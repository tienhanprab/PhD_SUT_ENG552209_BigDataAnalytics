# Iterative Engineering Loop for the QQQ Analytics Project

Use one stage at a time. Inspect the current repository and existing outputs before changing files.

## Master loop

```text
You are helping with a university Big Data Analytics project about QQQ technical indicators and next-day direction. Work on only the stage named below.

Loop: REQUIREMENT -> PLAN -> IMPLEMENT -> RUN -> INSPECT -> VALIDATE -> FIX -> DOCUMENT -> STOP.

Rules:
1. Read Big_Data_Analytics_Project_Assignment.pdf, README.md, config.json, and existing outputs first.
2. Preserve chronological order and prevent look-ahead leakage.
3. State the exact input, output, acceptance checks, and rubric item for this stage.
4. Make the smallest coherent change. Run it and inspect the resulting tables/charts.
5. If validation fails, diagnose and fix before moving on.
6. Record assumptions, commands, results, and remaining risks.
7. Stop after the stage acceptance criteria pass. Do not start the next stage.
8. Never invent measurements, outputs, or conclusions that were not produced by an actual run.

Current stage: [INSERT STAGE]
```

## Checklist protocol

- Begin every stage with all applicable items marked `[ ]`.
- Change an item to `[x]` only after citing evidence from an actual file, command, table, chart, or rendered artifact.
- Use `[N/A]` only with a written reason.
- If any critical item remains `[ ]`, the stage status is **FAIL** and the next stage must not begin.
- End every stage with changed files, commands run, observed results, remaining risks, and a **STOP/GO** decision.
- Keep this template unchanged; record completed checks in the stage report or commit rather than permanently pre-checking future work here.

---

# Stage 1 Requirement Analysis

ตรวจสอบ Requirement ของ Big Data Analytics Project และออกแบบหัวข้อ QQQ Technical Indicator Feature Importance

**Checklist:**

- [ ] Dataset มีอย่างน้อย 500 rows
- [ ] ระบุแหล่งข้อมูลชัดเจน
- [ ] มี Data Collection และ Data Cleaning
- [ ] มี EDA อย่างน้อย 5 chart types
- [ ] ใช้ ML หรือ Statistical Method อย่างน้อย 1 วิธี
- [ ] มี Research Questions อย่างน้อย 3 ข้อ
- [ ] มี Report 5–10 หน้า
- [ ] มี Presentation สำหรับ 10–15 นาที
- [ ] มี Code Notebook พร้อมคำอธิบาย
- [ ] สร้างตาราง mapping ระหว่าง requirement กับ deliverable

เมื่อเสร็จ ให้รายงาน Requirement ที่ผ่าน สิ่งที่ยังขาด และ Stop

---

# Stage 2 Project Setup

จัดเตรียมโครงสร้างโปรเจกต์ให้สามารถพัฒนาและรันซ้ำได้

**Checklist:**

- [ ] สร้างโฟลเดอร์ `data/raw` และ `data/processed`
- [ ] สร้างโฟลเดอร์ `src`, `outputs`, `tests` และ `deliverables`
- [ ] สร้าง `README.md`
- [ ] สร้าง `requirements.txt`
- [ ] สร้าง `config.json`
- [ ] กำหนด symbol, date range, random seed และ test fraction
- [ ] อธิบายคำสั่งติดตั้งและรันโปรเจกต์
- [ ] ตรวจว่าไม่มี API key หรือข้อมูลลับใน repository

เมื่อเสร็จ ให้แสดง Project Tree และ Stop

---

# Stage 3 Data Acquisition

พัฒนาโค้ดดึงข้อมูล QQQ Daily Historical Quotes จาก Nasdaq API

แหล่งข้อมูลอ้างอิง: <https://www.nasdaq.com/market-activity/etf/qqq/historical>

API endpoint: <https://api.nasdaq.com/api/quote/QQQ/historical>

**Checklist:**

- [ ] กำหนดช่วงวันที่จาก `config.json`
- [ ] ส่ง request พร้อม User-Agent
- [ ] ตรวจ HTTP status และ API response schema
- [ ] เก็บ response เดิมไว้ใน `data/raw`
- [ ] มีข้อมูลอย่างน้อย 500 rows
- [ ] บันทึก source URL และวันที่ดึงข้อมูล
- [ ] มีข้อความ error ที่เข้าใจง่ายเมื่อดาวน์โหลดไม่สำเร็จ
- [ ] ไม่แก้ไข raw data หลังดาวน์โหลด

เมื่อเสร็จ ให้รายงานจำนวนแถว ช่วงวันที่ และตำแหน่งไฟล์ raw แล้ว Stop

---

# Stage 4 Data Cleaning and Quality

ทำความสะอาดข้อมูล QQQ และสร้าง Data Quality Report

**Checklist:**

- [ ] แปลง Date เป็น datetime
- [ ] แปลง Open, High, Low และ Close เป็นตัวเลข
- [ ] แปลง Volume เป็นตัวเลข
- [ ] เรียงข้อมูลจากอดีตไปอนาคต
- [ ] ตรวจและลบ duplicate dates
- [ ] ตรวจ missing values
- [ ] ตรวจราคาติดลบหรือศูนย์
- [ ] ตรวจว่า High ≥ Open, Close และ Low
- [ ] ตรวจว่า Low ≤ Open, Close และ High
- [ ] สรุปจำนวนแถวก่อนและหลัง cleaning
- [ ] บันทึกผล Data Quality เป็น JSON หรือ CSV

ห้ามลบข้อมูลโดยไม่มีเหตุผลและคำอธิบาย เมื่อผ่าน checklist แล้ว Stop

---

# Stage 5 Feature Engineering

สร้าง Technical Indicators สำหรับใช้จำแนกทิศทางราคาวันถัดไป

Feature groups: Return history, Trend, Momentum, Volatility และ Volume

**Checklist:**

- [ ] สร้าง `return_1d` และ `return_5d`
- [ ] สร้าง opening gap
- [ ] สร้าง SMA และ EMA ratios
- [ ] สร้าง MACD และ MACD signal
- [ ] สร้าง RSI 14
- [ ] สร้าง Stochastic Oscillator
- [ ] สร้าง Rate of Change
- [ ] สร้าง ATR และ daily range
- [ ] สร้าง rolling volatility
- [ ] สร้าง Bollinger Band width
- [ ] สร้าง volume change และ volume ratio
- [ ] สร้าง `target_up` จาก Close(t+1) > Close(t)
- [ ] ตรวจว่า features ของวัน t ไม่ใช้ข้อมูลหลังวัน t
- [ ] ลบ warm-up rows อย่างมีเหตุผล
- [ ] บันทึก model-ready dataset

อธิบายสูตรและความหมายของแต่ละ feature group แล้ว Stop

---

# Stage 6 Exploratory Data Analysis

ทำ EDA และสร้างกราฟอย่างน้อย 5 ชนิด

**Checklist:**

- [ ] แสดง descriptive statistics
- [ ] Line chart ของราคา QQQ
- [ ] Histogram ของ daily returns
- [ ] Box plot เปรียบเทียบ return
- [ ] Scatter plot ระหว่าง indicator กับ next-day return
- [ ] Correlation heatmap
- [ ] Bar chart ของ target class balance
- [ ] ทุกกราฟมี title, labels และ units
- [ ] ตรวจ outliers และ missing values
- [ ] เขียน interpretation ใต้แต่ละกราฟ
- [ ] ไม่ใช้คำอธิบายเชิง causality หากข้อมูลแสดงเพียง correlation
- [ ] บันทึกกราฟใน `outputs/figures`

เมื่อเสร็จ ให้สรุป Insight จากแต่ละกราฟและ Stop

---

# Stage 7 Modeling

สร้างโมเดลจำแนกทิศทางราคาวันถัดไปโดยป้องกัน data leakage

Models: Majority-Class Baseline, Logistic Regression และ Random Forest Classifier

**Checklist:**

- [ ] แบ่ง Train/Test ตามลำดับเวลา
- [ ] ห้ามใช้ random train-test split
- [ ] ใช้ข้อมูลเก่าฝึกและข้อมูลใหม่ทดสอบ
- [ ] บันทึก train end date และ test start date
- [ ] ใช้ Pipeline สำหรับ preprocessing ที่จำเป็น
- [ ] กำหนด random seed
- [ ] ใช้ TimeSeriesSplit สำหรับ cross-validation
- [ ] จำกัด Random Forest เพื่อลด overfitting
- [ ] เปรียบเทียบกับ baseline
- [ ] บันทึก predictions และ probabilities

เมื่อเสร็จ ให้แสดงขนาด Train/Test และ model configuration แล้ว Stop

---

# Stage 8 Model Evaluation

ประเมินโมเดลบนข้อมูล Test ที่ไม่เคยเห็นมาก่อน

**Checklist:**

- [ ] Accuracy
- [ ] Balanced Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1 Score
- [ ] ROC AUC
- [ ] Confusion Matrix
- [ ] ROC Curve
- [ ] Time-series cross-validation mean
- [ ] Time-series cross-validation standard deviation
- [ ] เปรียบเทียบ Random Forest กับ baselines
- [ ] ตรวจว่าโมเดลดีกว่า chance หรือไม่
- [ ] ไม่สรุปว่าโมเดลดีจาก accuracy เพียงค่าเดียว

ให้ตอบอย่างตรงไปตรงมาหากโมเดลไม่ชนะ baseline แล้ว Stop

---

# Stage 9 Feature Importance and Insights

วิเคราะห์ว่าตัวแปรใดมี predictive association กับ next-day direction

**Checklist:**

- [ ] คำนวณ permutation importance บน Test set เท่านั้น
- [ ] ห้ามคำนวณ importance จากข้อมูล Train แล้วนำไปอ้างเป็นผลสุดท้าย
- [ ] แสดง importance mean
- [ ] แสดง importance standard deviation
- [ ] จัดอันดับ Top Features
- [ ] รวม importance ตาม feature family
- [ ] ตรวจ correlated features
- [ ] ระบุ feature ที่ importance ใกล้ศูนย์หรือติดลบ
- [ ] ไม่ใช้คำว่า cause, affect หรือ drive โดยไม่มี causal evidence
- [ ] ตอบ Research Question ทั้ง 3 ข้อด้วยค่าจริง

รูปแบบคำตอบ:

- RQ1: Top indicators และความไม่แน่นอน
- RQ2: Top feature family และข้อจำกัด
- RQ3: ผลการ generalize เทียบกับ baseline

เมื่อเสร็จ ให้แยก Findings, Interpretation และ Limitations แล้ว Stop

---

# Stage 10 Report

สร้างหรือปรับปรุงรายงานความยาว 5–10 หน้า

โครงสร้างรายงาน:

1. Title and Group Members
2. Introduction and Objectives
3. Data Description and Source
4. Data Cleaning Process
5. Exploratory Data Analysis
6. Analytical Methods
7. Results and Discussion
8. Conclusion and Recommendations
9. References

**Checklist:**

- [ ] ครบทุกหัวข้อตาม Assignment
- [ ] มี Research Questions อย่างน้อย 3 ข้อ
- [ ] มีกราฟอย่างน้อย 5 ประเภท
- [ ] ตัวเลขตรงกับ notebook และ output tables
- [ ] อ้างอิง Nasdaq
- [ ] อธิบาย chronological split
- [ ] อธิบาย data leakage prevention
- [ ] รายงานผลลัพธ์ที่ไม่ดีอย่างตรงไปตรงมา
- [ ] ระบุ limitations
- [ ] ความยาว 5–10 หน้า
- [ ] ตรวจทุกหน้าว่าไม่มีข้อความหรือกราฟถูกตัด

เมื่อผ่านทั้งหมด ให้รายงานจำนวนหน้าและ Stop

---

# Stage 11 Presentation

สร้าง Presentation สำหรับนำเสนอ 10–15 นาที

**Checklist:**

- [ ] Title และ Group Members
- [ ] Problem และ Research Questions
- [ ] Data Source และ Data Quality
- [ ] Cleaning และ Feature Engineering
- [ ] EDA Summary
- [ ] Model Design
- [ ] Model Evaluation
- [ ] Feature Importance
- [ ] คำตอบ Research Questions
- [ ] Conclusion และ Next Steps
- [ ] ตัวเลขตรงกับ report และ notebook
- [ ] กราฟหลักสามารถแก้ไขได้
- [ ] มี Speaker Notes
- [ ] เนื้อหาเหมาะสำหรับเวลา 10–15 นาที
- [ ] ไม่มีข้อความล้นหรือวัตถุซ้อนกัน

เมื่อผ่านแล้ว ให้สร้างสรุปเวลาพูดต่อสไลด์และ Stop

---

# Stage 12 Final Audit

ตรวจสอบโปรเจกต์ทั้งหมดก่อนส่ง

**Checklist:**

- [ ] รัน pipeline ตั้งแต่ต้นจนจบ
- [ ] Notebook รันทุก cell โดยไม่มี error
- [ ] Automated tests ผ่าน
- [ ] Dataset มีอย่างน้อย 500 rows
- [ ] EDA มีอย่างน้อย 5 chart types
- [ ] Research Questions ครบอย่างน้อย 3 ข้อ
- [ ] Report มี 5–10 หน้า
- [ ] Presentation เปิดได้
- [ ] ตัวเลขใน notebook, report และ slides ตรงกัน
- [ ] Source URLs เปิดได้
- [ ] ไม่มี API key หรือข้อมูลลับ
- [ ] ไม่มีไฟล์ชั่วคราวใน submission package
- [ ] ใส่ชื่อสมาชิกเรียบร้อย
- [ ] ระบุว่าเป็นงานเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน

ให้สรุปผลเป็นตาราง:

| Deliverable | Status | Evidence | Remaining Issue |
|---|---|---|---|

หากมีข้อใดไม่ผ่าน ให้แก้ไขและตรวจใหม่ ห้ามประกาศว่าเสร็จจนกว่า critical checklist จะผ่านทั้งหมด

---

## Review prompt for each iteration

```text
Act as a skeptical reviewer. Inspect the latest stage output and return: (1) pass/fail for every acceptance criterion, (2) evidence with file paths and values, (3) leakage or reproducibility risks, (4) the smallest required fixes, and (5) a stop/go decision for the next stage. Do not edit files.
```
