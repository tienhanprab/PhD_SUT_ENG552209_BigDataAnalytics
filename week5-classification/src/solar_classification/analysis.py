"""Reproducible chronological experiment and inspectable output tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from .download import download
from .models import OneR, Prism, TertileDiscretizer, majority

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ['light_lux', 'panel_temperature_c', 'voltage_v', 'current_a', 'power_w']


def run():
    raw = pd.read_csv(download())
    required = FEATURES + ['fault', 'timestamp_s', 'day', 'event_id', 'fault_type']
    if not set(required).issubset(raw.columns):
        raise ValueError('Unexpected source schema')
    if raw.fault.isna().any() or not set(raw.fault.unique()).issubset({0, 1}):
        raise ValueError('Missing/invalid class label')
    clean = raw.drop_duplicates().sort_values('timestamp_s').copy()
    if clean.timestamp_s.duplicated().any() or clean[['timestamp_s', 'day']].isna().any().any():
        raise ValueError('Ambiguous time ordering')
    clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    days = sorted(clean.day.unique())
    if len(days) != 3:
        raise ValueError('Pinned experiment expects exactly three simulated days')
    train, test = clean[clean.day < days[-1]].copy(), clean[clean.day == days[-1]].copy()
    train_events = set(train.loc[train.event_id.ne(0), 'event_id'])
    test_events = set(test.loc[test.event_id.ne(0), 'event_id'])
    if train_events & test_events:
        raise ValueError('An anomaly event crosses the split')
    assert train.timestamp_s.max() < test.timestamp_s.min()
    bins = TertileDiscretizer().fit(train[FEATURES])
    Xtrain, Xtest = bins.transform(train[FEATURES]), bins.transform(test[FEATURES])
    ytrain, ytest = train.fault.to_numpy(), test.fault.to_numpy()
    models = {'OneR': OneR().fit(Xtrain, ytrain), 'PRISM': Prism().fit(Xtrain, ytrain)}
    predictions = {'ZeroR': np.full(len(test), majority(ytrain))}
    predictions.update({name: model.predict(Xtest) for name, model in models.items()})
    tables = ROOT / 'outputs/tables'
    tables.mkdir(parents=True, exist_ok=True)
    metrics, per_class = [], []
    for name, pred in predictions.items():
        report = classification_report(ytest, pred, labels=[0, 1], output_dict=True, zero_division=0)
        rules = models[name].rules_ if name in models else []
        metrics.append(dict(model=name, accuracy=accuracy_score(ytest, pred),
                            balanced_accuracy=balanced_accuracy_score(ytest, pred),
                            macro_f1=report['macro avg']['f1-score'],
                            fault_precision=report['1']['precision'], fault_recall=report['1']['recall'],
                            fault_f1=report['1']['f1-score'], rules=len(rules),
                            total_conditions=sum(len(r.conditions) for r in rules),
                            mean_conditions=np.mean([len(r.conditions) for r in rules]) if rules else 0,
                            default_rules=1))
        for label in [0, 1]:
            per_class.append(dict(model=name, label=label, **report[str(label)]))
        pd.DataFrame(confusion_matrix(ytest, pred, labels=[0, 1]),
                     index=['actual_normal', 'actual_fault'],
                     columns=['pred_normal', 'pred_fault']).to_csv(tables / f'{name.lower()}_confusion_matrix.csv')
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(tables / 'model_comparison.csv', index=False)
    class_df = pd.DataFrame(per_class)
    class_df.to_csv(tables / 'per_class_metrics.csv', index=False)
    rules_records = []
    for name, model in models.items():
        for i, rule in enumerate(model.rules_, 1):
            mask = rule.mask(Xtest)
            n = int(mask.sum())
            rules_records.append(dict(model=name, rule_id=i, rule=rule.text(), label=rule.label,
                                      conditions=len(rule.conditions), train_support=rule.support,
                                      train_precision=rule.correct / rule.support,
                                      test_support=n,
                                      test_precision=float((ytest[mask] == rule.label).mean()) if n else np.nan))
    rules_df = pd.DataFrame(rules_records)
    rules_df.to_csv(tables / 'rules.csv', index=False)
    pd.DataFrame(models['OneR'].candidates_).to_csv(tables / 'oner_candidates.csv', index=False)
    balance = pd.crosstab(clean.day, clean.fault).rename(columns={0:'Normal',1:'Fault'})
    balance.to_csv(tables / 'class_balance_by_day.csv')
    edges = pd.DataFrame([dict(feature=f, cut_points=json.dumps(v.tolist()),
                               training_median=bins.medians_[f], intervals='right-closed; infinite tails')
                          for f, v in bins.cuts_.items()])
    edges.to_csv(tables / 'bin_edges.csv', index=False)
    _, unmatched, conflicts = models['PRISM'].predict_details(Xtest)
    combinations = Xtrain.assign(fault=ytrain).groupby(FEATURES).fault.agg(['size','nunique'])
    rejected = models['PRISM'].rejected_
    audit = dict(raw_rows=len(raw), raw_columns=len(raw.columns), clean_rows=len(clean),
                 duplicates_removed=len(raw)-len(clean), missing_by_column=raw.isna().sum().to_dict(),
                 train_rows=len(train), test_rows=len(test), train_days=[int(d) for d in days[:-1]],
                 test_days=[int(days[-1])], train_faults=int(ytrain.sum()), test_faults=int(ytest.sum()),
                 shared_fault_events=0, oner_feature=models['OneR'].feature_,
                 mixed_training_combinations=int((combinations['nunique'] > 1).sum()),
                 observed_training_combinations=len(combinations),
                 prism_rejected_candidates=len(rejected),
                 prism_deferred_positives=sum(r['positives'] for r in rejected),
                 prism_unmatched_test=int(unmatched.sum()), prism_conflicting_test=int(conflicts.sum()),
                 light_at_sensor_max_fraction=float((clean.light_lux == 65535).mean()),
                 power_vi_mae=float((clean.power_w-clean.voltage_v*clean.current_a).abs().mean()))
    (tables / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    pd.DataFrame(rejected).to_csv(tables / 'prism_rejected_candidates.csv', index=False)
    processed = ROOT / 'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    for name, data, X in [('train', train, Xtrain), ('test', test, Xtest)]:
        X.assign(fault=data.fault, source_row=data.index, timestamp_s=data.timestamp_s).to_csv(processed / f'{name}_discretized.csv', index=False)
    prediction_df = test[['timestamp_s', 'fault', 'fault_type']].copy()
    for name, pred in predictions.items():
        prediction_df[name] = pred
    prediction_df.to_csv(tables / 'test_predictions.csv', index=False)
    return dict(raw=raw, train=train, test=test, Xtrain=Xtrain, Xtest=Xtest, models=models,
                metrics=metric_df, per_class=class_df, rules=rules_df, edges=edges,
                balance=balance, audit=audit, predictions=predictions)


if __name__ == '__main__':
    result = run()
    print(result['metrics'].to_string(index=False))
    print(json.dumps(result['audit'], indent=2))
