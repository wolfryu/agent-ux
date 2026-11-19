# jmtt1-5k 新規 gold に対する regex / CPU モデル / regex+CPU の比較（2026-11-19）

## 1. 目的

新しく作成した typed gold `data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl` を使い、
既存の軽量な 3 構成を**同一データ・同一評価条件**で比較する。

- A: `rules`（regex + 辞書）
- B: `model`（CPU モデル単体）
- C: `cpu`（regex + 辞書 + CPU モデル）

既存の `sumi-ja-test` 200 件の結果とは、文書ドメインもラベル分布も異なる。
本レポートの数値を 200 件の数値と直接混ぜて比較してはいけない。

---

## 2. 評価データ

| 項目 | 値 |
|---|---:|
| 文書数 | 798 |
| span 数 | 10,238 |
| ユニークエンティティ数 | 7,967 |
| span がある文書 | 689 |
| span がない文書 | 109 |
| 未解決エンティティ | 2 |
| フィルタされたエンティティ | 20 |

- 原文: `参考用旧项目/files/jmtt/jmtt1-5k`（医療 Wikipedia 記事、1〜5KB）
- gold: DeepSeek-V4.1-Flash-1017 による **typed + offset** の silver/gold
- 人手の二重注釈ではない。臨床記録でもない。

### ラベル分布

| label | spans | 割合 |
|---|---:|---:|
| ORGANIZATION | 7,669 | 74.9% |
| PERSON | 1,555 | 15.2% |
| ADDRESS | 923 | 9.0% |
| ACCOUNT | 32 | 0.31% |
| URL | 21 | 0.21% |
| DOB | 19 | 0.19% |
| POSTAL_CODE | 16 | 0.16% |
| AGE | 3 | 0.03% |

> `DATE` は gold に含めていない（設立年・発表日・承認日などは個人情報ではないため）。
> `ACCOUNT` は銀行口座だけでなく、SNS/オンラインアカウントも含む。
> `POSTAL_CODE` は `〒` を含む場合がある。`ADDRESS` は郵便番号を含まない。
> 同一エンティティの全出現箇所を span として展開している。

---

## 3. 比較した 3 構成

| 構成 | tier | detector 構成 | モデル |
|---|---|---|---|
| A: regex | `rules` | `RuleDetector` + `GazetteerDetector` | なし |
| B: CPU モデル単体 | `model` | `HFTagger` のみ | `NagaYu/sumi-ja-pii`（ModernBERT-ja 130M, ONNX int8） |
| C: regex+CPU | `cpu` | `RuleDetector` + `GazetteerDetector` + `HFTagger` | 同上 |

- ローカルモデル: `models/sumi-ja-pii/model.int8.onnx`（128MB）
- tokenizer/config: `models/sumi-ja-pii/`
- `model` は「モデル単体」の ablation、`cpu` は「regex + モデル」の本番候補に相当する。

### 実行時の注意（実際に起きたこと）

最初に `--onnx` を付けずに `model` を実行したところ、`HFTagger` が PyTorch backend を選び、
venv に `torch` がないため `ModuleNotFoundError: No module named 'torch'` で失敗した。
以下では**ローカル ONNX を明示**して再実行した。

```text
--model models/sumi-ja-pii
--onnx  models/sumi-ja-pii/model.int8.onnx
```

---

## 4. 実行コマンド

```bash
# A: regex
PYTHONPATH=src .venv/Scripts/python.exe -m piimask eval \
  --data data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl \
  --detector rules \
  --report reports/20261119-jmtt-regex.json

# B: CPU モデル単体
PYTHONPATH=src .venv/Scripts/python.exe -m piimask eval \
  --data data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl \
  --detector model \
  --model models/sumi-ja-pii \
  --onnx models/sumi-ja-pii/model.int8.onnx \
  --report reports/20261119-jmtt-model.json

# C: regex + CPU
PYTHONPATH=src .venv/Scripts/python.exe -m piimask eval \
  --data data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl \
  --detector cpu \
  --model models/sumi-ja-pii \
  --onnx models/sumi-ja-pii/model.int8.onnx \
  --report reports/20261119-jmtt-regex-cpu.json
```

保存先:

- `reports/20261119-jmtt-regex.json`
- `reports/20261119-jmtt-model.json`
- `reports/20261119-jmtt-regex-cpu.json`
- 実行ログ: `out/jmtt_eval_rules.log`, `out/jmtt_eval_model2.log`, `out/jmtt_eval_cpu.log`

---

## 5. 評価指標

`piimask eval` の定義に従う。

- **P / R / F1**: 同一ラベル・部分一致（overlap ratio ≥ 0.5）の one-to-one match
- **critical recall**: critical ラベルの recall（本データでは 10,235 span が対象）
- **miss rate**: `1 - critical recall`
- **over-mask rate**: gold と全く重ならない予測の割合
- **char IoU**: 予測文字集合と gold 文字集合の IoU
- **gate**: critical recall ≥ 0.995 を合格とする

本データの gold に存在するラベルは 8 種類:
`ORGANIZATION`, `PERSON`, `ADDRESS`, `ACCOUNT`, `URL`, `DOB`, `POSTAL_CODE`, `AGE`。
`DATE`, `EMAIL`, `PHONE`, `MYNUMBER`, `ROOM`, `IP_ADDRESS` は gold に存在しないため、
P/R/F1 の採点対象外（`annotated_labels_only`）である。
ただし予測された場合は `unannotated_predictions` として別途記録される。

---

## 6. 全体結果

| 構成 | 全予測 | 採点対象予測 | P | R | F1 | critical R | miss | over-mask | char IoU | 採点外予測 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: regex | 5,622 | 4,992 | **0.4621** | 0.2253 | 0.3030 | 0.2251 | 0.7749 | 0.4232 | 0.2428 | 630 |
| B: CPU モデル単体 | 3,728 | 3,275 | 0.4275 | 0.1367 | 0.2072 | 0.1368 | 0.8632 | 0.4195 | 0.1897 | 453 |
| C: regex+CPU | 8,690 | 7,673 | 0.4443 | **0.3330** | **0.3807** | **0.3328** | **0.6672** | 0.4328 | **0.2710** | 1,017 |
| H: BERT（**sumi 200 で学習 → jmtt 798 でテスト**）<br>`phi-deidentification-bert` smoke / expanded 8-label<br>公式モデルの重みなし、リークなし | 28,278 | 27,720 | 0.0839 | 0.2272 | 0.1226 | 0.2273 | 0.7727 | 0.5320 | 0.1958 | 558 |

- **F1 は C (0.3807) > A (0.3030) > B (0.2072)**。
- C は A より recall を +0.108 改善するが、precision は −0.018 低下し、over-mask は +0.010 増える。
- B は単体では最も弱い。特に ORGANIZATION が 0 件、EMAIL/DOB の誤予測が多い。
- 3 構成とも critical recall gate（0.995）は **FAIL**。
- **H の注記**: H は `sumi-ja-test` 200 篇だけで学習し、jmtt1-5k 798 篇でテストした **リークなしの cross-domain smoke**。jmtt のデータは一切学習に使っていない。ただし公式 BERT の重みではなく、`ORGANIZATION` の学習例が sumi にないため ORG recall は 0.005 とほぼ全滅。H は「sumi で学習した BERT は jmtt にそのまま移らない」ことの参考行であり、A/B/C/Gemma の順位に混ぜない。詳細は第 12 節。

### 全ラベル表示（未注釈ラベルを含む参考値）

| 構成 | P | R | F1 |
|---|---:|---:|---:|
| A: regex | 0.4104 | 0.2253 | 0.2909 |
| B: CPU モデル単体 | 0.3755 | 0.1367 | 0.2005 |
| C: regex+CPU | 0.3923 | 0.3330 | 0.3602 |

---

## 7. ラベル別結果

`—` は gold の support が 0 のため採点対象外。

| label | support | A P | A R | A F1 | B P | B R | B F1 | C P | C R | C F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ORGANIZATION | 7,669 | 0.579 | 0.228 | 0.327 | 0.000 | 0.000 | 0.000 | 0.579 | 0.228 | 0.327 |
| PERSON | 1,555 | 0.453 | 0.015 | 0.030 | 0.467 | 0.576 | 0.516 | 0.467 | 0.567 | 0.512 |
| ADDRESS | 923 | 0.284 | 0.547 | 0.374 | 0.628 | 0.524 | 0.571 | 0.349 | 0.810 | 0.487 |
| ACCOUNT | 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| URL | 21 | 0.518 | 0.667 | 0.583 | 0.000 | 0.000 | 0.000 | 0.518 | 0.667 | 0.583 |
| DOB | 19 | 0.000 | 0.000 | 0.000 | 0.009 | 0.210 | 0.017 | 0.003 | 0.053 | 0.005 |
| POSTAL_CODE | 16 | 1.000 | 0.938 | 0.968 | 0.400 | 1.000 | 0.571 | 0.400 | 1.000 | 0.571 |
| AGE | 3 | 0.030 | 1.000 | 0.059 | 0.000 | 0.000 | 0.000 | 0.030 | 1.000 | 0.059 |
| DATE | 0 | — | — | — | — | — | — | — | — | — |
| EMAIL | 0 | — | — | — | — | — | — | — | — | — |
| PHONE | 0 | — | — | — | — | — | — | — | — | — |
| MYNUMBER | 0 | — | — | — | — | — | — | — | — | — |
| ROOM | 0 | — | — | — | — | — | — | — | — | — |
| IP_ADDRESS | 0 | — | — | — | — | — | — | — | — | — |

---

## 8. 考察

### 8.1 最大のボトルネックは ORGANIZATION

- gold 10,238 span のうち **7,669 span（74.9%）** が ORGANIZATION。
- A（rules）は 3,018 件予測して recall **0.228**、precision **0.579**。
- B（CPU モデル単体）は ORGANIZATION を **1 件も出さない**（recall 0）。
- C（regex+CPU）も A と同じ 0.228 / 0.579。モデル側に ORGANIZATION の出力がないため、
  実質的に rules の gazetteer だけが働いている。

Wikipedia の組織名は「株式会社」「病院」「大学」などの接尾辞を持たないもの、
長い複合名、英語名、略称が多く、現在の gazetteer の suffix 依存では届かない。
`sumi-ja-pii` の学習分布にも ORGANIZATION が十分含まれていない可能性がある。

### 8.2 PERSON はモデルだけが実用的

- A は敬称・姓名フィールド依存のため recall **0.015**。
- B は recall **0.576** / precision **0.467**。
- C は recall **0.567** / precision **0.467** で、B とほぼ同等。

Wikipedia の人物名は敬称なしが大半なので、regex の PERSON はほぼ機能しない。
モデルは一定の recall を出すが、precision は 5 割弱で、過剰検出も多い。

### 8.3 ADDRESS は C で recall が上がるが precision が落ちる

| 構成 | P | R | F1 |
|---|---:|---:|---:|
| A | 0.284 | 0.547 | 0.374 |
| B | **0.628** | 0.524 | **0.571** |
| C | 0.349 | **0.810** | 0.487 |

C は recall を 0.547 → 0.810 に伸ばすが、precision は 0.284 → 0.349 程度。
予測数は 2,146 件で、gold 923 件に対して過剰。
住所の境界（番地・建物名・ビル名）と、住所でない地名の混入が課題。

### 8.4 POSTAL_CODE は rules が最もバランス良い

- A: precision **1.000** / recall **0.938**
- B: precision 0.400 / recall **1.000**
- C: precision 0.400 / recall **1.000**

B/C は 16 件の gold に対して 40 件予測しており、数字列を郵便番号として過剰検出している。
rules の `〒NNN-NNNN` アンカーは非常に安定している。

### 8.5 URL は rules のみ

- A/C: precision 0.518 / recall 0.667
- B: 0

モデルは URL を検出しない。URL は rules の正規表現に任せるべき。

### 8.6 DOB / AGE は fusion と過剰検出の問題が大きい

- DOB:
  - A: 0
  - B: precision 0.009 / recall 0.210
  - C: precision 0.003 / recall 0.053
- AGE:
  - A/C: precision 0.030 / recall 1.000（3 件の gold に対して 99 件予測）
  - B: 0

モデルは一般の日付を DOB と誤認しやすく、C では rules の DATE/AGE が優先されて
モデルの DOB を押し出している可能性がある。AGE は `歳` の広い一致で過剰マスクになっている。

### 8.7 ACCOUNT（SNS ハンドル）は全滅

- gold 32 件（`@handle` や Facebook/Instagram の裸 handle）。
- A: 0 件
- B: 96 件予測 / TP 0
- C: 77 件予測 / TP 0

モデルは ACCOUNT ラベル自体は出すが、SNS ハンドルを学習していないため TP にならない。
`@handle`、`Facebook`、`Instagram`、`X`、`YouTube` の文脈を見る専用ルールが必要。

### 8.8 採点外予測と over-mask

採点外予測は A 630 件、B 453 件、C 1,017 件。
主な内訳は DATE（A 618 件）、EMAIL（B 423 件 / C 360 件）、
MYNUMBER、PHONE、ROOM、IP_ADDRESS など。

これらは P/R/F1 からは除外されるが、実運用では**過剰マスク**として問題になる。
特に 109 件の「PII なし」文書では、1 件でも予測すると pure FP になる。
over-mask rate が 42〜43% と高いのは、この影響が大きい。

### 8.9 C は A より良いが、まだ実用には遠い

- F1: C 0.3807 > A 0.3030 > B 0.2072
- recall: C 0.3330 > A 0.2253 > B 0.1367
- precision: A 0.4621 > C 0.4443 > B 0.4275
- char IoU: C 0.2710 > A 0.2428 > B 0.1897

C は recall/F1/IoU で A を上回るが、precision はわずかに低下し、over-mask は増える。
特に DOB は B 0.210 → C 0.053 と悪化しており、fusion の仲裁ロジックを見直す必要がある。
（rules の DATE/AGE がモデルの DOB を上書きしている可能性が高い。）

---

## 9. 制約

1. **gold は LLM 生成の silver/gold** であり、人手の二重注釈ではない。
2. 原文は医療 Wikipedia 記事で、臨床記録ではない。
3. 109 件は PII なし。2 件は未解決、20 件はフィルタ済み。
4. `DATE` は gold に含めていない。`ACCOUNT` は SNS を含む。
5. `sumi-ja-test` 200 件とはラベル分布が大きく異なるため、数値を直接比較できない。
6. PHONE / EMAIL / MYNUMBER / ROOM / IP_ADDRESS は gold に存在しない。
7. 本レポートの数値は、この新データに対する**ドメイン適合性の確認**であり、
   臨床現場の受入判定ではない。

---

## 10. 結論と次のアクション

### 結論

- 3 構成の中では **C: regex+CPU が最も良い**（F1 0.3807 / recall 0.3330）。
- ただし critical recall 0.3328 で、gate 0.995 には遠い。
- 最大の原因は **ORGANIZATION（gold の 74.9%）を実質検出できていないこと**。
- 次に PERSON の precision、ADDRESS の境界、DOB/AGE の過剰検出、ACCOUNT（SNS）の未対応。

### 次のアクション候補

1. **ORGANIZATION 専用の gazetteer / モデル**を追加する
   （Wikipedia の組織名、英語名、略称、接尾辞なし組織）。
2. **fusion の仲裁**を見直す
   （rules が model の DOB/ADDRESS を不必要に上書きしない）。
3. **SNS ハンドル → ACCOUNT** の専用ルールを追加する。
4. **DATE/AGE の過剰マスク**を抑える
   （一般日付と生年月日、年齢と期間の分離）。
5. 同じ新 gold に対して **LLM 系（E/F/G）** も評価し、軽量層との差を確認する。
6. gold の一部を人手で監査し、silver のノイズ（未解決 2 件、フィルタ 20 件）を確認する。
7. 最終的な臨床受入は、別途 200〜500 件の実臨床データ二重注釈で行う。

---

## 付録: 出力ファイル

- `reports/20261119-jmtt-regex.json`
- `reports/20261119-jmtt-model.json`
- `reports/20261119-jmtt-regex-cpu.json`
- `out/jmtt_eval_rules.log`
- `out/jmtt_eval_model2.log`
- `out/jmtt_eval_cpu.log`
- gold: `data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl`
- gold の説明: `data/private/jmtt1-5k-gold/README.md`

---

## 11. 追加実験: ローカル Gemma 4 12B（llm-only-jmtt）

> この節は既存の A/B/C の結果を変更せず、末尾に追加したものです。

### 11.1 目的と設定

同じ 798 篇の gold に対して、ローカルで起動した **Gemma 4 12B QAT** を
LLM 単体（`llm-only-jmtt`）で評価した。

| 項目 | 設定 |
|---|---|
| モデル | `google/gemma-4-12b-qat`（LM Studio ローカル） |
| API | `http://127.0.0.1:1234/v1` |
| 推論 | `llm-only-jmtt`（ORGANIZATION / URL / AGE を含むフルラベル） |
| reasoning | `reasoning_effort=none`（Gemma 4 は `/no_think` だけでは不十分） |
| 呼び出し | 1 文書 1 call、`max_chars=4000`、`max_calls=1` |
| キャッシュ | `out/20261119-gemma-jmtt-cache.json` |
| レポート | `reports/20261119-jmtt-gemma4.json` |

実行コマンド:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m piimask eval \
  --data data/private/jmtt1-5k-gold/jmtt1-5k-gold.jsonl \
  --detector llm-only-jmtt \
  --llm-model google/gemma-4-12b-qat \
  --llm-max-calls 1 \
  --llm-max-chars 4000 \
  --llm-cache out/20261119-gemma-jmtt-cache.json \
  --report reports/20261119-jmtt-gemma4.json
```

### 11.2 全体結果

| 構成 | 全予測 | P | R | F1 | critical R | miss | over-mask | char IoU | 採点外 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: regex | 5,622 | 0.4621 | 0.2253 | 0.3030 | 0.2251 | 0.7749 | 0.4232 | 0.2428 | 630 |
| B: CPU モデル単体 | 3,728 | 0.4275 | 0.1367 | 0.2072 | 0.1368 | 0.8632 | 0.4195 | 0.1897 | 453 |
| C: regex+CPU | 8,690 | 0.4443 | 0.3330 | 0.3807 | 0.3328 | 0.6672 | 0.4328 | 0.2710 | 1,017 |
| **D: Gemma 4 12B（LLM 単体）** | **8,604** | **0.9188** | **0.7721** | **0.8391** | **0.7721** | **0.2279** | **0.0732** | **0.7403** | **0** |

- LLM usage: `calls=798`, `chars_sent=773,379`, `cache_hits=0`, `dropped_invalid=0`。
- **F1 は D 0.8391 ≫ C 0.3807 > A 0.3030 > B 0.2072**。
- precision 0.9188、recall 0.7721、over-mask 0.0732、char IoU 0.7403 と、
  軽量 3 構成より大幅に良い。
- ただし critical recall 0.7721 で、gate 0.995 は **FAIL**。

### 11.3 ラベル別結果（Gemma 4）

| label | support | pred | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| ORGANIZATION | 7,669 | 6,304 | 0.916 | 0.753 | 0.826 |
| PERSON | 1,555 | 1,445 | 0.965 | 0.896 | 0.929 |
| ADDRESS | 923 | 760 | 0.880 | 0.725 | 0.795 |
| ACCOUNT | 32 | 25 | 1.000 | 0.781 | 0.877 |
| URL | 21 | 42 | 0.500 | 1.000 | 0.667 |
| DOB | 19 | 8 | 0.875 | 0.368 | 0.518 |
| POSTAL_CODE | 16 | 16 | 0.875 | 0.875 | 0.875 |
| AGE | 3 | 4 | 0.750 | 1.000 | 0.857 |

### 11.4 考察

- **ORGANIZATION が大幅改善**: F1 0.826（C は 0.327）。gold の 74.9% を占める最大のボトルネックを、
  LLM がほぼ実用レベルまで検出した。
- **PERSON も強い**: F1 0.929（C 0.512）。敬称なし・英語名・ふりがなもかなり拾えている。
- **ADDRESS**: F1 0.795（C 0.487）。ただし recall 0.725 で、住所の一部欠落は残る。
- **ACCOUNT（SNS ハンドル）**: F1 0.877。`@handle` 系は LLM の方が圧倒的に強い。
- **URL**: recall 1.000 / precision 0.500。42 件予測して 21 件が TP。
  公式サイト以外のリンクや引用 URL も拾っている可能性がある。
- **DOB は弱い**: recall 0.368。一般の日付を DOB と誤認しない代わりに、
  生年月日の取りこぼしが増えている。
- **POSTAL_CODE / AGE**: 高精度。件数は少ないが F1 0.875 / 0.857。
- **over-mask 0.0732**: 軽量層の 0.42〜0.43 から大幅改善。109 件の PII なし文書でも
  予測がかなり抑えられている。
- **char IoU 0.7403**: 境界も軽量層より大幅に良い。

### 11.5 制約

- これは **LLM 単体**の結果であり、rules/CPU との cascade ではない。
- gold は LLM 生成の silver/gold。人手の二重注釈ではない。
- ローカル 12B の 1 call/doc、`reasoning_effort=none` の結果。
- `URL` の precision 0.500、`DOB` の recall 0.368 はまだ改善余地が大きい。
- 本結果は既存 A/B/C を置き換えるものではなく、末尾に追加した参考比較である。

### 11.6 追加ファイル

- `reports/20261119-jmtt-gemma4.json`
- `out/jmtt_eval_gemma4.log`
- `out/20261119-gemma-jmtt-cache.json`
- `src/piimask/detectors/llm_verifier.py` の `LLM_ONLY_JMTT_SYSTEM_PROMPT`
- `src/piimask/pipeline.py` の `llm-only-jmtt` tier


---
## 12. 追加: phi-deidentification-bert（sumi 200 で学習 → jmtt 798 でテスト）

> **この節はリークなしの cross-domain smoke 評価である。**
> BERT は `sumi-ja-test` 200 篇だけで学習し、`jmtt1-5k` 798 篇は**一切学習に使っていない**。
> 以前の jmtt 797 篇で学習した in-sample 結果は削除済み。
> ただし公式 `sociocom/phi-deidentification-bert` の重みではなく、
> 公開 `train.py` で作った smoke checkpoint である。

### 12.1 設定

| 項目 | 内容 |
|---|---|
| ベースモデル | `sbintuitions/modernbert-ja-310m` |
| 学習データ | `sumi-ja-test` **200 篇のみ** |
| 学習 | 1 epoch、max_length 1024、batch 2 |
| 学習時 eval | sumi の 20 篇、token-level F1 **0.9045**（近似ラベル） |
| checkpoint | `external/phi-deidentification-bert/out_sumi_smoke/` |
| テストデータ | `jmtt1-5k` **798 篇**（学習と完全不重複） |
| テスト推論 | `transformers` pipeline、GPU、直接 span 抽出 |
| 生 span | `out/phi_bert_jmtt_spans_sumi_trained.jsonl`（798 篇、18,219 span） |
| 評価 | `scripts/eval_phi_bert_jmtt.py` |
| レポート | `reports/20261119-phi-bert-jmtt-sumi-trained.json` |

- 学習データとテストデータの重複は **0 篇**。
- tagged text ではなく pipeline から直接 span を抽出しているため、offset 整合性は **0 問題**。

### 12.2 ラベル対応

| BERT ラベル | jmtt への対応 | 備考 |
|---|---|---|
| `phi_person` | PERSON | 学習あり（sumi 811） |
| `phi_hospital` | ORGANIZATION | **sumi に学習例なし** |
| `phi_location` | ADDRESS + POSTAL_CODE | 1 対多（duplicate） |
| `phi_id` | ACCOUNT + URL | 1 対多（duplicate） |
| `phi_age` | DOB + AGE | 1 対多（duplicate） |
| `phi_tel` | PHONE | jmtt gold に PHONE 0 |
| `phi_job` | JOB | jmtt taxonomy 外のため破棄 |

### 12.3 全体結果

| 見方 | 対象ラベル | 予測数 | P | R | F1 | critical R | over-mask | char IoU | support |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict | PERSON / ORGANIZATION / ADDRESS / DOB | 13,898 | **0.1628** | **0.2226** | **0.1881** | 0.2226 | 0.4899 | 0.1855 | 10,166 |
| expanded | 8 ラベル（duplicate あり） | 28,278 | 0.0839 | 0.2272 | 0.1226 | 0.2273 | 0.5320 | 0.1958 | 10,238 |
| agnostic | ラベル無視（any PHI） | 28,278 | 0.1971 | **0.5445** | 0.2895 | 0.5445 | 0.5320 | 0.1958 | 10,238 |

- **sumi で学習した BERT は jmtt にほぼ移らない**。
- expanded F1 **0.1226**、strict F1 **0.1881**、agnostic F1 **0.2895**。
- over-mask **0.5320**。予測 28,278 件の半分以上がどの gold とも重ならない。
- char IoU **0.1958**。境界も大きく崩れている。

### 12.4 ラベル別結果（expanded 8 ラベル）

| label | support | pred | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| ORGANIZATION | 7,669 | 341 | 0.103 | **0.005** | **0.009** |
| PERSON | 1,555 | 6,475 | 0.219 | **0.912** | 0.353 |
| ADDRESS | 923 | 5,736 | 0.138 | 0.856 | 0.237 |
| ACCOUNT | 32 | 3,370 | 0.008 | 0.844 | 0.016 |
| URL | 21 | 3,370 | 0.006 | 0.952 | 0.012 |
| DOB | 19 | 1,346 | 0.014 | 1.000 | 0.028 |
| POSTAL_CODE | 16 | 5,736 | 0.003 | 1.000 | 0.006 |
| AGE | 3 | 1,346 | 0.000 | 0.000 | 0.000 |
| PHONE | 0 | 558 | — | — | — |

- **ORGANIZATION はほぼ全滅**: recall 0.005（7,669 件中 35 TP）。
  sumi に ORGANIZATION の学習例がなく、`phi_hospital` を学習できていない。
- **PERSON は recall 0.912 だが precision 0.219**。人名らしきものを大量に出すが、
  jmtt の組織名・住所・一般名詞も PERSON に寄せている。
- **ADDRESS / POSTAL_CODE は recall 0.856 / 1.000、precision 0.138 / 0.003**。
  `phi_location` の duplicate と過剰検出で FP が膨らむ。
- **ACCOUNT / URL / DOB も recall は高いが precision 0.006〜0.014**。
  `phi_id` / `phi_age` の duplicate により、ほぼ全予測が FP になっている。
- **AGE 0/3**。`phi_age` は DOB 側に寄っており、AGE を区別できていない。

### 12.5 既存結果との並置（順位ではない）

| 手法 | 学習データ | テスト | ラベル | P | R | F1 | 備考 |
|---|---|---|---:|---:|---:|---:|---|
| A regex | なし | jmtt 798 | 8 | 0.4621 | 0.2253 | 0.3030 | 既存 |
| B CPU モデル | sumi 系 | jmtt 798 | 8 | 0.4275 | 0.1367 | 0.2072 | 既存 |
| C regex+CPU | なし | jmtt 798 | 8 | 0.4443 | 0.3330 | 0.3807 | 既存 |
| Gemma 4 12B | なし | jmtt 798 | 8 | 0.9188 | 0.7721 | 0.8391 | 既存 |
| **H BERT strict** | **sumi 200** | **jmtt 798** | **4** | 0.1628 | 0.2226 | 0.1881 | **リークなし cross-domain** |
| **H BERT expanded** | **sumi 200** | **jmtt 798** | **8（duplicate）** | 0.0839 | 0.2272 | 0.1226 | **リークなし cross-domain** |
| **H BERT agnostic** | **sumi 200** | **jmtt 798** | **any** | 0.1971 | 0.5445 | 0.2895 | **リークなし cross-domain** |

- H は A/B/C/Gemma より大幅に低い。**sumi の一般文書で学習した BERT は、
  jmtt の組織名中心の Wikipedia 記事にそのまま適用できない**ことを示す。
- ただし H は公式 BERT ではなく smoke checkpoint であり、
  sumi に ORGANIZATION の学習例がないというデータ制約も大きい。
- したがって H を「BERT の実力」と一般化せず、**cross-domain smoke の下限**として扱う。
- A/B/C/Gemma の順位表には混ぜない。

### 12.6 制約

1. 公式 `sociocom/phi-deidentification-bert` の重みではない。
2. 学習は sumi 200 篇のみ。jmtt 798 篇はテスト専用で、リークはない。
3. sumi に ORGANIZATION / JOB の学習例がないため、jmtt の最大ラベル ORG を学習できない。
4. `phi_location` / `phi_id` / `phi_age` の duplicate により FP が大きく出る。
5. これは「sumi→jmtt」の cross-domain 下限であり、BERT 一般の性能ではない。

### 12.7 関連ファイル

- `reports/20261119-phi-bert-jmtt-sumi-trained.json`
- `out/phi_bert_jmtt_spans_sumi_trained.jsonl`
- `out/phi_bert_extract_jmtt_sumi_trained.log`
- `out/phi_bert_eval_jmtt_sumi_trained.log`
- `out/phi_bert_train_sumi_200.log`
- `external/phi-deidentification-bert/configs/sumi_train_200.yaml`
- `external/phi-deidentification-bert/out_sumi_smoke/`
- `docs/phi-deidentification-bert-integration.md`


