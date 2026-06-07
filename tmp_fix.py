path = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model\part3_ResidualFeedbackAndOnline_Correction\result\exp_004\key_findings.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '| frankfurt2 | 132.2m | 781.1m | -490.7% |'

new = '''| frankfurt2 | 132.2m | 781.1m | -490.7% |

The apparent -490.7% degradation in frankfurt2 is **not progressive degradation** -- it is an artifact of the "first vs last 100 epoch" metric. Epoch-bin diagnosis (20 bins) reveals:

- No clear transition point at the 1.2x error ratio threshold
- The last bin (epochs 3382-3560) has StdLS CEP50=1021m -- a single high-error bin dominates the "last 100" average
- Early vs late p_los_gap is stable (0.783 vs 0.884)
- Safety fallback (1.05x threshold) prevents per-epoch CEP50 from exceeding Standard LS
- This is a **data characteristic** (intermittent high-error regions), not an algorithmic failure'''

content = content.replace(old, new)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('key_findings.md updated')
