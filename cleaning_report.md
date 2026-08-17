# AI Data Cleaning Engine - Report
_Generated: 2026-08-17 12:25:19_

## Summary
- Rows before cleaning: **56135**
- Rows after cleaning: **55001**
- Rows removed (duplicates/etc.): **1134**
- Missing values before: **7929**
- Missing values after: **16610**

## Missing Values (Before Cleaning)
| column            |   missing_count |   missing_pct | dtype   |
|:------------------|----------------:|--------------:|:--------|
| student_id        |              35 |          0.06 | str     |
| student_name      |              35 |          0.06 | str     |
| age               |              35 |          0.06 | float64 |
| major             |            1476 |          2.63 | str     |
| year              |            1613 |          2.87 | str     |
| city              |            1511 |          2.69 | str     |
| enrollment_date   |              35 |          0.06 | str     |
| gpa               |              35 |          0.06 | str     |
| last_grade        |            1444 |          2.57 | str     |
| semester_fee      |              35 |          0.06 | str     |
| enrollment_status |              35 |          0.06 | str     |
| email             |              35 |          0.06 | str     |
| phone             |            1500 |          2.67 | str     |
| credits_completed |              35 |          0.06 | float64 |
| attendance_pct    |              35 |          0.06 | float64 |
| has_scholarship   |              35 |          0.06 | str     |

## Duplicates (Before Cleaning)
- Exact duplicate rows found: 1134 (2.02%)

## Outliers (Before Cleaning, Z-score method)
| column            |   outlier_count |   outlier_pct |
|:------------------|----------------:|--------------:|
| age               |             359 |          0.64 |
| credits_completed |               0 |          0    |
| attendance_pct    |            1069 |          1.9  |

## ML Outlier Detection (Isolation Forest)
- Rows flagged as anomalies: 1100

## Missing Values Left As-Is (After Cleaning)
These columns still contain missing values **on purpose** -- there's no reliable way to guess a missing email, phone number, or similar identifying detail, so the engine leaves them missing rather than fabricating a plausible-looking fake value:

- **student_id**: 1 missing (0.0%)
- **student_name**: 1 missing (0.0%)
- **major**: 2218 missing (4.03%)
- **year**: 2283 missing (4.15%)
- **city**: 2182 missing (3.97%)
- **enrollment_date**: 1 missing (0.0%)
- **last_grade**: 2128 missing (3.87%)
- **enrollment_status**: 1 missing (0.0%)
- **email**: 5654 missing (10.28%)
- **phone**: 2140 missing (3.89%)
- **has_scholarship**: 1 missing (0.0%)
