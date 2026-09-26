# Interpretable Classification of Solar Photovoltaic Faults Using OneR and PRISM

โปรเจกต์ **Week 5: Classification Rule Induction** ใช้ Kaggle Solar PV Anomaly Detection Dataset
version 2 พร้อมเขียน OneR และ PRISM เอง, notebook ที่รันซ้ำได้ และ HTML สำหรับอ่านผล

## เปิดผลงาน

- [Notebook พร้อมผลรัน](notebooks/solar_pv_classification.ipynb)
- [HTML preview ของ notebook](report/solar_pv_classification.html)
- [ข้อกำหนดจากบทเรียนและ design choices](docs/ASSIGNMENT_REQUIREMENTS.md)
- [Data dictionary และความต่างจากคำอธิบาย Kaggle](docs/DATA_DICTIONARY.md)

## ผลจริง

ข้อมูล 152,973 แถว × 16 คอลัมน์ ไม่มี missing หรือ exact duplicates ใช้ 5 measured features:
`light_lux`, `panel_temperature_c`, `voltage_v`, `current_a`, `power_w` และ target `fault`.
**Lux ไม่ใช่ irradiance**; ไม่ใช้ `true_*`, `fault_type`, IDs หรือเวลาเป็น predictor.

Train = วัน 0–1 จำนวน 101,279 แถว; test = วัน 2 จำนวน 51,694 แถว (Fault 4,422).
แบ่งเวลาก่อน fit median/quantile bins เพื่อป้องกัน leakage; anomaly event ไม่ข้าม split.

| Model | Test accuracy | Fault precision | Fault recall | Fault F1 | Learned rules |
|---|---:|---:|---:|---:|---:|
| ZeroR | 91.45% | 0.00%* | 0.00% | 0.0000 | 0 + default |
| OneR | 91.45% | 0.00%* | 0.00% | 0.0000 | 2 |
| PRISM | 91.90% | 100.00% | 5.36% | 0.1017 | 12 |

\* ไม่มีการทำนาย Fault จึงกำหนด undefined precision เป็น 0. แต่ละโมเดลมี default rule
แยกจาก learned rules; precision/recall/F1 **ทั้งสอง classes** อยู่ใน notebook และ CSV.

OneR เลือก `light_lux` จาก tie ตามลำดับ features: ทุก attribute มี training error เท่ากัน
และกฎทำนาย Normal ทั้งหมด จึง **ไม่ใช่หลักฐานว่า light เป็น feature ที่ดีที่สุด**.
PRISM ตรวจพบ 237 จาก 4,422 fault rows และพลาด 4,185 แถว แม้ accuracy สูง
จึงยังไม่เหมาะสรุปว่าเป็นระบบตรวจจับ fault ที่มีประสิทธิภาพ.

## วิธีรัน

```bash
cd week5-classification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m solar_classification.download
python -m solar_classification.analysis
python tests/test_models.py
python -m solar_classification.notebook
MPLCONFIGDIR="$PWD/.matplotlib" python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 notebooks/solar_pv_classification.ipynb
python -m jupyter nbconvert --to html --output-dir report notebooks/solar_pv_classification.ipynb
```

Tests ใช้ standard-library unittest ได้โดยไม่ต้องติดตั้ง pytest.
ตัวดาวน์โหลด pin version 2 และตรวจ SHA-256 ทุกครั้ง; raw CSV มีอยู่ในเครื่องแต่ไม่ commit.
`requirements-lock.txt` บันทึก versions ของ dependencies หลักที่ใช้รันครั้งนี้.

## Pipeline และอัลกอริทึม

Kaggle → schema/quality audit → chronological split → train-only imputation/tertiles
→ handwritten OneR + PRISM + ZeroR → held-out metrics → confusion matrices → rules.

- OneR เลือก attribute ที่ total training error ต่ำสุด; tie class → 0, tie feature → column order.
- PRISM เลือกเงื่อนไขจาก p/t สูงสุด, tie → positive coverage, feature/category order.
- PRISM เก็บเฉพาะ pure rules. เมื่อใช้ครบ attributes แล้วยังปน class ให้ reject candidate
  และ defer positives โดยคง negatives ไว้; เป็น noise/termination policy ที่ระบุเพิ่มจาก lecture.
- No-match หรือ conflicting-class match ใช้ training majority. Test มี no-match 31,038 แถว
  และ conflict 0 แถว; default จึงมีบทบาทสูง. กฎต่างข้ออาจ cover แถวซ้ำกัน.
- Tertile ไม่ใช่ supervised bucket-size discretization บนหน้า 18 ของ lecture.
  ถ้า quantiles ซ้ำ/ชนขอบให้ลดจำนวน bins; `light_lux` มีสอง bins เพราะ sensor saturation.
- ไม่มี tuning ด้วย test set. ไม่มี class balancing หรือสมมติผลล่วงหน้า.

## โครงสร้าง

```text
src/solar_classification/  # download, models, analysis, notebook builder
notebooks/                # executed teaching notebook
report/                   # HTML notebook preview
outputs/tables/           # metrics ทั้งสอง classes, rules, bins, predictions, audit
outputs/figures/          # class balance, confusion matrices, comparison
data/raw/                # local CSV + provenance (CSV ignored by Git)
data/processed/           # train/test categorical data, generated locally
docs/                    # source metadata, assignment mapping, dictionary
tests/                   # lecture example + algorithm edge cases
```

## ข้อจำกัด

[Dataset](https://www.kaggle.com/datasets/isurumy93/solar-pv-anomaly-detection-dataset/versions/2)
เป็น **simulated data ไม่ใช่ field measurements**; มีเพียงสามวันจากระบบเดียวและแถวติดกันสัมพันธ์กัน.
Class imbalance, sensor saturation, redundant power inputs และ bins หยาบจำกัดการตรวจ fault.
กฎเป็นความสัมพันธ์ในข้อมูล ไม่ยืนยันสาเหตุทางกายภาพ. การปรับ bins หรือ noise-tolerant PRISM
ควรเป็นงานทดลองใหม่โดยมี validation ภายใน train ก่อน ไม่ปรับตาม test วัน 2.
