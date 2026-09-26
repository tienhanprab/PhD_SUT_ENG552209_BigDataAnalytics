import numpy as np
import pandas as pd
from solar_classification.models import OneR, Prism, TertileDiscretizer


def test_oner_selects_minimum_error_and_unseen_default():
    X = pd.DataFrame({'noise':['a','a','a','a'], 'signal':['l','l','h','h']})
    y = np.array([0,0,1,1])
    model = OneR().fit(X,y)
    assert model.feature_ == 'signal'
    assert np.array_equal(model.predict(X), y)
    assert model.predict(pd.DataFrame({'signal':['unseen']}))[0] == 0


def test_prism_conjunction_and_pure_rules():
    X = pd.DataFrame({'a':['l','l','h','h'], 'b':['l','h','l','h']})
    y = np.array([0,0,0,1])
    model = Prism().fit(X,y)
    assert np.array_equal(model.predict(X),y)
    assert any(len(r.conditions)==2 and r.label==1 for r in model.rules_)
    assert all(r.correct == r.support for r in model.rules_)


def test_prism_contradictions_terminate_and_default():
    X = pd.DataFrame({'a':['l','l','h']})
    model = Prism().fit(X,np.array([0,1,0]))
    assert len(model.rejected_) == 2
    pred, unmatched, conflicts = model.predict_details(pd.DataFrame({'a':['l','unknown']}))
    assert pred.tolist() == [0,0]
    assert unmatched.all() and not conflicts.any()


def test_discretizer_uses_only_training_and_handles_ties():
    train = pd.DataFrame({'x':[0.,1.,2.,3.,4.,5.,6.], 'constant':[2.]*7})
    bins = TertileDiscretizer().fit(train)
    saved = bins.cuts_['x'].copy()
    out = bins.transform(pd.DataFrame({'x':[-999.,999.,np.nan,2.], 'constant':[2.,3.,np.nan,2.]}))
    assert out.x.tolist() == ['Low','High','Medium','Low']
    assert out.constant.tolist() == ['Medium']*4
    assert np.array_equal(saved,bins.cuts_['x'])


def test_weather_lecture_minimum_error():
    # Lecture p.3: Outlook is the minimum-error attribute (4/14).
    rows = [('sunny','hot','high',0,0),('sunny','hot','high',1,0),
            ('cloudy','hot','high',0,1),('rainy','mild','high',0,1),
            ('rainy','cool','normal',0,1),('rainy','cool','normal',1,0),
            ('cloudy','cool','normal',1,1),('sunny','mild','high',0,0),
            ('sunny','cool','normal',0,1),('rainy','mild','normal',0,1),
            ('sunny','mild','normal',1,1),('cloudy','mild','high',1,1),
            ('cloudy','hot','normal',0,1),('rainy','mild','high',1,0)]
    df = pd.DataFrame(rows,columns=['outlook','temperature','humidity','windy','play'])
    X,y=df.drop(columns='play'),df.play.to_numpy()
    one=OneR().fit(X,y)
    assert one.feature_ == 'outlook'
    assert (one.predict(X)!=y).sum() == 4
    prism=Prism().fit(X,y)
    assert np.array_equal(prism.predict(X),y)


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in list(globals().items()) if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
