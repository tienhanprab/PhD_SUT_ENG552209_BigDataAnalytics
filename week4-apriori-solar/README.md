# Association Rule Mining of Solar Photovoltaic Power Generation

โปรเจกต์ Week 4 สำหรับทำ Association Rule Mining ด้วย **Apriori** จาก
[Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
บน Kaggle โดยแปลงข้อมูล numeric ของโรงไฟฟ้าพลังงานแสงอาทิตย์เป็น transaction lists
เช่น:

```text
[
  "Irradiation_High",
  "AmbientTemp_High",
  "ModuleTemp_High",
  "ACPower_High"
]
```

โปรเจกต์นี้ใช้ข้อมูลจริงทั้ง 4 CSV, ตรวจ checksum, aggregate generation จากระดับ
inverter เป็นระดับ plant-time ก่อน exact-join กับ weather sensor และไม่สมมติ rules
ล่วงหน้า

## Assignment coverage

1. เลือก Kaggle dataset: Solar Power Generation Data version 1
2. เตรียมข้อมูลเป็น transaction lists
3. ใช้ Apriori และ tune `min_support`/`min_confidence`
4. สร้าง Top-10 rules table และ directed network graph
5. ตีความ 5 rules ด้วยบริบท photovoltaic พร้อม support, confidence และ lift

รายละเอียดว่าส่วนใดเป็นข้อกำหนดจาก lecture และส่วนใดเป็น design choice ของโปรเจกต์อยู่ใน
[`docs/ASSIGNMENT_REQUIREMENTS.md`](docs/ASSIGNMENT_REQUIREMENTS.md)

## Project structure

```text
week4-apriori-solar/
├── config.json
├── pyproject.toml
├── src/solar_apriori/
│   ├── analysis.py
│   ├── download.py
│   └── notebook.py
├── tests/
├── data/
│   ├── raw/                  # four pinned Kaggle CSVs (not committed)
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── tables/
├── report/
└── docs/
```

## Environment setup

ใช้ Python 3.12:

```bash
cd week4-apriori-solar
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Download and validate the dataset

```bash
download-solar-data
```

ตัวดาวน์โหลด pin dataset version 1 และตรวจ filename, byte size, SHA-256 และ schema
ของทั้ง 4 ไฟล์ก่อนนำเข้า `data/raw/`. ถ้าไฟล์ถูกต้องอยู่แล้วจะ reuse โดยไม่ดาวน์โหลดซ้ำ
และจะสร้าง `solar_power_generation.metadata.json` สำหรับ provenance. Kaggle credentials
ถูกจัดการโดย KaggleHub และไม่ถูกเก็บในโปรเจกต์

ข้อมูลต้นทางถูกระบุโดย Kaggle ว่า `Data files © Original Authors`; โปรเจกต์จึงไม่ควร
redistribute หรือ commit CSV ต้นฉบับ

## Build and execute the notebook

```bash
build-solar-apriori-notebook

MPLCONFIGDIR="$PWD/.matplotlib" python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/solar_pv_apriori.ipynb

MPLCONFIGDIR="$PWD/.matplotlib" python -m jupyter nbconvert \
  --to html \
  --output-dir report \
  notebooks/solar_pv_apriori.ipynb
```

Notebook ที่ execute แล้วอยู่ที่
[`notebooks/solar_pv_apriori.ipynb`](notebooks/solar_pv_apriori.ipynb) และ HTML preview
อยู่ที่ [`report/solar_pv_apriori.html`](report/solar_pv_apriori.html)

## Method summary

- Transaction grain: หนึ่ง `(PLANT_ID, DATE_TIME)` ต่อ 15 นาที
- Generation measure: ค่า `AC_POWER` เฉลี่ยต่อ inverter record ที่สังเกตได้
- Primary population: daylight (`IRRADIATION > 0`) และ inverter coverage ≥ 50%
- Temporal split: discovery 24 วันแรก; validation 10 วันท้าย
- Discretization: plant-specific tertiles fit จาก discovery period เท่านั้น
- Transaction items: irradiation, ambient temperature, module temperature และ AC power
- Rule direction: environmental antecedent 1-2 items → AC-power consequent 1 item
- Tuning grid: support 0.02-0.25 และ confidence 0.50-0.80
- Selected thresholds: `min_support=0.075`, `min_confidence=0.60`, `max_len=3`
- Stability filters: ≥30 joint observations, ≥5 discovery dates, lift >1.1 ใน discovery,
  lift >1 ใน validation และในแต่ละ plant

`DC_POWER`, cumulative yields, IDs และ timestamps ไม่ถูกใช้เป็น rule items เพื่อหลีกเลี่ยง
กฎเชิง tautology หรือ proxy ที่ไม่ตอบคำถามหลัก

## Validated results

- Exact generation-weather matches: **6,416 plant-time rows**
- Final daylight transactions: **3,607**
- Discovery: **2,527 transactions / 24 days**
- Validation: **1,080 transactions / 10 days**
- Selected run: 58 frequent itemsets, 83 confidence-qualified rules,
  19 environmental→AC-power rules, 19 stable rules และ 10 nonredundant rules

ตัวอย่างผลสำคัญจากทุก 3,607 transactions:

| Rule | Support | Confidence | Lift |
|---|---:|---:|---:|
| High ambient temperature + High irradiation → High AC power | 10.98% | 87.42% | 2.97 |
| High irradiation → High AC power | 24.18% | 82.81% | 2.81 |
| Low irradiation → Low AC power | 33.60% | 96.11% | 2.77 |
| Low ambient temperature + Medium irradiation → Medium AC power | 12.12% | 87.75% | 2.45 |
| Medium irradiation + Medium module temperature → Medium AC power | 22.18% | 84.57% | 2.36 |

ผลเหล่านี้เป็น association ณ timestamp เดียวกัน ไม่ใช่ causal effect หรือ forecast.
Low/Medium/High เป็นระดับสัมพัทธ์ภายในแต่ละ plant และ support เป็นสัดส่วนของ daylight
plant-time intervals ไม่ใช่สัดส่วนพลังงานหรือจำนวนวัน

## Main outputs

- `data/processed/transactions.csv`
- `outputs/tables/bin_edges.csv`
- `outputs/tables/threshold_tuning.csv`
- `outputs/tables/top_10_rules.csv`
- `outputs/figures/threshold_tuning_heatmap.png`
- `outputs/figures/top_10_rule_network.png`

