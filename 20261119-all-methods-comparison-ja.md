# 日本語 PII 検出：全手法の単独成績比較（2026-11-19）

- データ: `data/public/sumi-ja-test.jsonl` 全 **200 文書 / 1,633 span / 93,372 字**
- 評価: entity-level partial match（同ラベル、重なり ≥ 0.5、1 対 1）、`scope=annotated`、bootstrap 500、`enable_guard=False`
- 注: 本ドキュメントは**各手法の単独成績のみ**を比較する。2 手法の組合せは行わない。組合せ C 自体を 1 手法として掲載する。
- 詳細な実験条件と手法別の失敗分析: `20261119-regexcpu-result.md` の第 7 / 13 / 14 / 15 / 16 / 17 章

## 0. 結論（要約）

- **総合 F1 が最高: C（0.9796）**。次いで E（0.9726）、F（0.9697）。
- **再現率が最高: B（0.9945）**。0.995 ゲートまで **1 TP** に迫る。C は 0.9859。
- **適合率が最高: E（0.9899）**。F 0.9885、C 0.9734。
- **ローカル 8 ラベルの高精度: F（Gemma 4 12B、F1 0.9697）**。ただし ACCOUNT の再現率 0.7547 が弱点。
- **G（qwen3-swallow、thinking-off）は再現率が最低（0.6901）**で、推奨しない。reasoning 有効版は未実行。
- **0.995 ゲートを通過した手法はない**。B が最も近く、C / E / F が続く。

## 1. 実験条件と評価口径（先に読む）

| 項目 | 内容 |
|---|---|
| データ | `sumi-ja-test.jsonl` 200 文書（`--limit` なし） |
| gold | 1,633 span、**8 ラベル**: PERSON / PHONE / EMAIL / ADDRESS / POSTAL_CODE / DOB / ACCOUNT / MYNUMBER |
| 未注釈 | DATE / ORGANIZATION / PATIENT_ID / ROOM / FAX（gold 0） |
| マッチング | partial match（同ラベル、重なり比 ≥ 0.5、1 対 1 の貪欲マッチ） |
| 採点範囲 | gold に support がある 8 ラベルのみ。未注釈ラベルへの予測は別集計 |
| critical recall | 8 ラベルはすべて critical のため、critical recall = overall recall |
| 速度 | A/B/C/D-v1/D-v2 は `bench`、F/G は eval の wall time、E は再開実行の参考値。**条件が完全には同じでないため、厳密な順位付けは不可** |
| データの性質 | `sumi-ja-pii` の学習データと同源の一般文書。**スコアは分布内の上界であり、臨床精度ではない** |

> **表を読む際の注意**: A/C の 615 個の未注釈予測（DATE/ORG/PATIENT_ID/ROOM/FAX）は、annotated の P/R では**採点されない**が、`over-mask`、`char IoU`、`all-label precision` には加算される。したがって A/C の over-mask/IoU を「臨床テキストを壊した量」と直接読んではいけない。B/D-v2/E/F/G の未注釈予測が 0 なのは、表が「きれい」に見えるだけで、19 ラベルをカバーしているという意味ではない。

## 2. 全体成績

### 2.1 precision / recall / F1 / critical recall / gate

| 手法 | P | R | F1 | critical R (95%CI) | gate 0.995 との差 |
|---|---:|---:|---:|---|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.9573 | 0.5083 | 0.6640 | 0.5083 (0.4648–0.5187) | 0.4867 |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 0.9160 | 0.9945 | 0.9536 | 0.9945 (0.9919–0.9982) | 0.0005 |
| C regex + CPU<br>A + B の融合 | 0.9734 | 0.9859 | 0.9796 | 0.9859 (0.9790–0.9906) | 0.0091 |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 0.9443 | 0.7679 | 0.8470 | 0.7679 (0.7505–0.8101) | 0.2271 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 0.8239 | 0.8365 | 0.8301 | 0.8365 (0.7993–0.8485) | 0.1585 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 0.9899 | 0.9559 | 0.9726 | 0.9559 (0.9353–0.9635) | 0.0391 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 0.9885 | 0.9516 | 0.9697 | 0.9516 (0.9268–0.9613) | 0.0434 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 0.9118 | 0.6901 | 0.7856 | 0.6901 (0.6551–0.7270) | 0.3049 |
| H BERT（ローカル smoke）<br>`phi-deidentification-bert` / expanded 8-label<br>公式モデルの重みなし | 0.3814 | 0.6840 | 0.4897 | 0.6840 (0.6602–0.7108) | 0.3110 |
| I MedPHINER-Llama-3.1-Swallow-8B<br>native `<phi_*>` prompt / expanded 8-label<br>200 docs / 54 incomplete | 0.6042 | 0.7244 | 0.6589 | 0.7244 (0.7134–0.7726) | 0.2706 |

### 2.2 over-mask / IoU / 予測数 / 未注釈 / all-label precision

| 手法 | over-mask | char IoU | annotated 予測 | 未注釈予測 | 総予測 | all-label P |
|---|---:|---:|---:|---:|---:|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.4150 | 0.4857 | 867 | 615 | 1,482 | 0.5601 |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 0.0039 | 0.9424 | 1,773 | 0 | 1,773 | 0.9160 |
| C regex + CPU<br>A + B の融合 | 0.2732 | 0.6537 | 1,654 | 615 | 2,269 | 0.7096 |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 0.1536 | 0.7504 | 1,328 | 176 | 1,504 | 0.8338 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 0.1743 | 0.7431 | 1,658 | 0 | 1,658 | 0.8239 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 0.0101 | 0.9727 | 1,577 | 0 | 1,577 | 0.9899 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 0.0102 | 0.9640 | 1,572 | 0 | 1,572 | 0.9885 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 0.0850 | 0.7276 | 1,236 | 0 | 1,236 | 0.9118 |
| H BERT（ローカル smoke）<br>`phi-deidentification-bert` / expanded 8-label<br>公式モデルの重みなし | 0.0882 | 0.5170 | 2,929 | 530 | 3,459 | 0.3229 |
| I MedPHINER-Llama-3.1-Swallow-8B<br>native `<phi_*>` prompt / expanded 8-label<br>200 docs / 54 incomplete | 0.3639 | 0.5221 | 1,958 | 122 | 2,080 | 0.5687 |

- `over-mask` は「どの gold とも重ならない予測 / 全予測」。A/C の 615 個の未注釈予測が値を押し上げる。B の 0.0039 は「B が 8 ラベルしか出さない」ためであり、**B のマスクがより安全という意味ではない**。
- `char IoU` も未注釈予測を予測側の和集合に含む。A/C の低値には DATE/ORG など未注釈ラベルの予測（**正しさは未検証**）が含まれる。F/E の 0.96–0.97 は主に 8 ラベルの境界品質による。**IoU は E 0.9727 が最高、次いで F 0.9640、B 0.9424。**
- `all-label P` は未注釈予測を FP として数える。A 0.5601、C 0.7096 は「8 ラベルの適合率」ではない。8 ラベルの適合率は 2.1 を参照。
- **H の注記**: H は公式 `sociocom/phi-deidentification-bert` の重みではなく、公開 `train.py` と jmtt1-5k の 797 篇で学習した smoke checkpoint である。`expanded` は `phi_location → ADDRESS/POSTAL_CODE`、`phi_id → EMAIL/ACCOUNT/MYNUMBER` の duplicate を含むため、**A–G と同じ 8 ラベル比較ではない**。H の 2.1/2.2 は参考行であり、順位付けには使わない。詳細は第 12 節。
- **I の注記**: I は `MedPHINER-Llama-3.1-Swallow-8B` を **native `<phi_*>` prompt** で評価したもので、Gemma 4 の JSON prompt とは異なる（MedPHINER は JSON prompt を守れないため、同一 prompt では 0 件になる）。sumi 200 篇中 **54 篇で本文が不完全**（モデルが途中で出力を止める）ため、recall は下限である。`phi_location → ADDRESS/POSTAL_CODE`、`phi_id → ACCOUNT/MYNUMBER`、`phi_age → DOB` の duplicate を含む。I は A–G の順位に混ぜない。詳細は第 13 節。

## 3. ラベル別成績（8 ラベル）

### 3.1 Recall（再現率）

| 手法 | PERSON | PHONE | EMAIL | ADDRESS | POSTAL_CODE | DOB | ACCOUNT | MYNUMBER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.1159 | 0.8345 | 1.0000 | 0.8929 | 0.9362 | 0.8171 | 0.9811 | 0.8182 |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 0.9988 | 0.9894 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9057 | 1.0000 |
| C regex + CPU<br>A + B の融合 | 0.9988 | 0.9331 | 1.0000 | 0.9929 | 1.0000 | 0.9756 | 1.0000 | 1.0000 |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 0.7460 | 0.8592 | 0.9430 | 0.9214 | 0.2553 | 0.7439 | 0.6226 | 0.8182 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 0.7374 | 0.9577 | 1.0000 | 0.9714 | 0.6915 | 0.9878 | 0.8491 | 1.0000 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 0.9211 | 0.9965 | 0.9810 | 1.0000 | 1.0000 | 1.0000 | 0.9245 | 1.0000 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 0.9260 | 0.9965 | 1.0000 | 1.0000 | 0.9574 | 1.0000 | 0.7547 | 0.9091 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 0.5931 | 0.8662 | 0.8797 | 0.7643 | 0.4255 | 0.8415 | 0.6415 | 1.0000 |

### 3.2 Precision（適合率）

| 手法 | PERSON | PHONE | EMAIL | ADDRESS | POSTAL_CODE | DOB | ACCOUNT | MYNUMBER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.9038 | 0.9793 | 1.0000 | 0.8929 | 0.9263 | 1.0000 | 1.0000 | 1.0000 |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 0.9939 | 0.9929 | 0.6529 | 0.9929 | 1.0000 | 0.9647 | 0.4752 | 0.9167 |
| C regex + CPU<br>A + B の融合 | 0.9818 | 0.9815 | 1.0000 | 0.9026 | 0.9307 | 0.9756 | 1.0000 | 1.0000 |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 0.9853 | 0.9242 | 0.9933 | 0.9627 | 0.8571 | 0.9385 | 0.5238 | 0.9000 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 0.9836 | 0.9347 | 1.0000 | 0.9784 | 0.9848 | 0.3913 | 0.2542 | 0.9167 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 0.9842 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9535 | 1.0000 | 1.0000 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 0.9804 | 0.9930 | 1.0000 | 1.0000 | 1.0000 | 0.9880 | 1.0000 | 1.0000 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 0.9959 | 0.9609 | 1.0000 | 0.9907 | 1.0000 | 0.8313 | 0.2957 | 0.9167 |

### 3.3 F1

| 手法 | PERSON | PHONE | EMAIL | ADDRESS | POSTAL_CODE | DOB | ACCOUNT | MYNUMBER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.2055 | 0.9011 | 1.0000 | 0.8929 | 0.9312 | 0.8993 | 0.9905 | 0.9000 |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 0.9963 | 0.9912 | 0.7900 | 0.9964 | 1.0000 | 0.9820 | 0.6234 | 0.9565 |
| C regex + CPU<br>A + B の融合 | 0.9902 | 0.9567 | 1.0000 | 0.9456 | 0.9641 | 0.9756 | 1.0000 | 1.0000 |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 0.8491 | 0.8905 | 0.9675 | 0.9416 | 0.3934 | 0.8299 | 0.5690 | 0.8571 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 0.8428 | 0.9461 | 1.0000 | 0.9749 | 0.8125 | 0.5606 | 0.3913 | 0.9565 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 0.9516 | 0.9982 | 0.9904 | 1.0000 | 1.0000 | 0.9762 | 0.9608 | 1.0000 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 0.9524 | 0.9947 | 1.0000 | 1.0000 | 0.9783 | 0.9939 | 0.8602 | 0.9524 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 0.7434 | 0.9111 | 0.9360 | 0.8629 | 0.5970 | 0.8364 | 0.4048 | 0.9565 |

> この 3 表は gold がある 8 ラベルのみ。DATE / ORGANIZATION / PATIENT_ID / ROOM / FAX は gold がないため、ラベル別成績は算出できない。A/C の未注釈ラベルへの予測は第 4 節を参照。

## 4. カバレッジと未注釈予測

| 手法 | カバー可能なラベル範囲（taxonomy） | 未注釈予測（採点外） |
|---|---|---:|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 19 ラベル（ルール + 辞書） | 615（DATE 354 / ORG 154 / PATIENT_ID 79 / ROOM 17 / FAX 11） |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 8 ラベル | 0 |
| C regex + CPU<br>A + B の融合 | 19 ラベル（A + B） | 615（A と同じ） |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 19 ラベル許可（v1 prompt） | 176 |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 8 ラベル（v2 prompt） | 0 |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 8 ラベル（v2 prompt） | 0 |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 8 ラベル（v2 prompt） | 0 |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 8 ラベル（v2 prompt） | 0 |

- **B / D-v2 / E / F / G は 8 ラベルのみ**。DATE / ORGANIZATION / PATIENT_ID / ROOM / FAX を出さないため、19 ラベル PII ツールとしてのカバレッジは不完全である。
- **A / C のルール・辞書層は 19 ラベルをカバーできる**。この corpus では実際に 13 ラベル（8 annotated + 5 unannotated）が出現した。615 個の予測は DATE/ORG/患者 ID/ROOM/FAX のマスク候補だが、gold がないため**正しさは未検証で、採点できない**。
- D-v1 の v1 prompt は 19 ラベルを許可し、未注釈予測は 176。D-v2 は 8 ラベルに絞ったため未注釈予測は 0。

## 5. 速度とリソース

| 手法 | 200 文書の合計 | 1 文書あたり | 字/秒 | 測定条件 |
|---|---:|---:|---:|---|
| A regex<br>`RuleDetector` + `GazetteerDetector` | 0.10 s | 0.5 ms | 961,970 | `bench`（warmup 後 200 文書） |
| B CPU モデル単独<br>`sumi-ja-pii` int8 ONNX | 33.67 s | 168.4 ms | 2,773 | `bench`（warmup 後 200 文書） |
| C regex + CPU<br>A + B の融合 | 39.18 s | 195.9 ms | 2,383 | `bench`（warmup 後 200 文書） |
| D-v1 ローカル Qwen3-8B<br>v1 prompt（医療・19 ラベル・残余） | 228.17 s | 1,140.9 ms | 409 | `bench`（warmup 後 200 文書） |
| D-v2 ローカル Qwen3-8B<br>v2 prompt（一般文書・8 ラベル） | 237.43 s | 1,187.1 ms | 393 | `bench`（warmup 後 200 文書） |
| E 商用 LLM<br>DeepSeek-V4.1-Flash-1017 / v2 | 未測定（参考: 99 calls / 100 s） | 約 1.0 s/call（参考） | 未計算 | eval の再開実行（101 cache + 99 calls）。初回 101 calls は stall を含む |
| F ローカル Gemma 4 12B QAT<br>v2 / reasoning_effort=none | 317 s | 1,585 ms | 294 | eval の wall time（200 calls、cache 0） |
| G ローカル qwen3-swallow-8B<br>v2 / thinking-off | 401 s | 2,005 ms | 233 | eval の wall time（200 calls、cache 0） |

- A/B/C/D-v1/D-v2 は同一マシンの `bench`（warmup 後 200 文書）で、比較可能性が高い。
- F/G は eval の wall time（200 calls、cache 0）であり、独立した bench ではない。A/B/C の `bench` と完全に同じ条件ではない。
- E にはクリーンな 200-call bench がない。初回 101 件で 1 回 API 呼び出しが stall し、停止後に cache から再開した。レポートの 100 秒は **99 calls + 101 cache hits** の再開時間であり、200 calls の総時間ではない。E の「約 1.0 s/call」は参考値にすぎない。
- 速度順（概ね比較可能な A/B/C/D/F/G 内）: A > B > C > D-v1 > D-v2 > F > G。E は厳密な順位付けから除外する。

## 6. データ出所

| 手法 | 結果 JSON | prompt / リソース |
|---|---|---|
| A | `reports/20261119-regex.json` | ルール + 辞書、LLM なし |
| B | `reports/20261119-model.json` | `NagaYu/sumi-ja-pii` int8 ONNX、rules なし |
| C | `reports/20261119-regex-cpu.json` | A + B |
| D-v1 | `reports/20261119-llm.json` | ローカル `qwen/qwen3-8b`、v1 prompt（医療・19 ラベル） |
| D-v2 | `reports/20261119-llm-v2.json` | ローカル `qwen/qwen3-8b`、v2 prompt（一般・8 ラベル） |
| E | `reports/20261119-llm-api.json` | 商用 `deepseek-ai/DeepSeek-V4.1-Flash-1017`、v2 |
| F | `reports/20261119-llm-gemma.json` | ローカル `google/gemma-4-12b-qat`、v2、`reasoning_effort=none` |
| G | `reports/20261119-llm-swallow.json` | ローカル `qwen3-swallow-8b-rl-v0.2`、v2、thinking-off |

## 7. 利点・欠点の総表

> 表中の単一値は Recall。ペアは R/P の順（P/R と明記した箇所を除く）。正確な値は第 3 章を参照。

| 手法 | 主な利点 | 主な欠点 | 適する用途 |
|---|---|---|---|
| **A regex** | 形式が閉じた型に強い：EMAIL 1.000/1.000、ACCOUNT P 1.000/R 0.981、POSTAL P 0.926/R 0.936、DOB P 1.000、MYNUMBER P 1.000。0.5 ms/文書。完全に決定的で監査可能 | PERSON R 0.116。PHONE R 0.835、DOB R 0.817、MYNUMBER R 0.818。全体 R 0.508。文脈の意味を扱えない。615 個の未注釈予測が all-label P/over-mask/IoU を悪化させる | 形式検証、guard、B/C の土台。**単独の PII ツールには不適** |
| **B CPU モデル** | **R 0.9945 が最高**、IoU 0.9424（A/B/C/D 内で最高；全手法では E 0.9727 / F 0.9640 が上回る）。PERSON 0.999、POSTAL/DOB 1.000、PHONE 0.989。168.4 ms。未注釈 0。gate まで 1 TP | EMAIL P 0.653（84 FP）、ACCOUNT P 0.475（53 FP）。149 FP。8 ラベルのみで DATE/ORG/患者 ID/ROOM/FAX なし。チェックディジット/構造検証なし | **再現率優先の神経層**。ルールと組み合わせて EMAIL/ACCOUNT/19 ラベルを補う |
| **C regex+CPU** | **F1 0.9796 が最高**、P 0.9734。19 ラベル。EMAIL/ACCOUNT/MYNUMBER 1.000。PERSON F1 0.990。196 ms。ルールが B の断片化を修正 | R 0.9859 < B 0.9945。ルール優先の融合が正しいモデル span を覆う：PHONE/FAX 11、POSTAL/PHONE 7、DATE/DOB 2、ROOM/ADDRESS 1。ADDRESS P 0.903、POSTAL P 0.931、PERSON P 0.982。over-mask/IoU は 615 個の未注釈予測の影響 | **on-prem の単一ツールとして第一候補**（19 ラベル + 最高 F1）。fusion の修正が必要 |
| **D-v1 ローカル Qwen3-8B（v1）** | P 0.9443 / F1 0.8470。PERSON 0.746、ADDRESS 0.921、EMAIL 0.943。ローカル 1.14 s。19 ラベル許可（176 未注釈） | POSTAL R 0.255、DOB R 0.744、ACCOUNT P 0.524/R 0.623、MYNUMBER R 0.818。PERSON R 0.746。70 dropped。v1 prompt は医療・残余向け | 歴史的ベースライン。「prompt だけ変えても不十分」の証拠 |
| **D-v2 ローカル Qwen3-8B（v2）** | R 0.8365。POSTAL R 0.692、PHONE R 0.958、EMAIL 1.000、ADDRESS 0.971、MYNUMBER 1.000。未注釈 0。dropped 13。1.19 s | **DOB P 0.391（126 FP）、ACCOUNT P 0.254（132 FP）**。PERSON R 0.737。F1 0.8301 < D-v1 0.8470。over-mask 0.174。8 ラベル | v2 prompt のベースライン。8B では能力不足で、v2 は E/F でこそ機能する |
| **E 商用 LLM** | **P 0.9899 が最高**、R 0.9559、F1 0.9726。PERSON R 0.921、PHONE R 0.997/P 1.000、EMAIL R 0.981/P 1.000、ADDRESS/POSTAL R 1.000/P 1.000、DOB R 1.000/P 0.954、ACCOUNT P 1.000/R 0.925、MYNUMBER R 1.000/P 1.000。16 FP/72 FN | 外部 API。1 回 stall 後に cache から再開（99 calls + 101 cache）で、**クリーンな 200-call bench がない**。PERSON 64 FN、ACCOUNT 4、EMAIL 3（`@sub.example.test`）、PHONE 1。gate 差 0.0391。費用/DPA/リージョン。8 ラベル | **外部送信が許される場合の高精度案** |
| **F ローカル Gemma 4 12B** | P 0.9885 / R 0.9516 / F1 0.9697。PERSON R 0.926、PHONE R 0.997/P 0.993、EMAIL/ADDRESS R 1.000/P 1.000、POSTAL R 0.957/P 1.000、DOB R 1.000/P 0.988、ACCOUNT P 1.000/R 0.755、MYNUMBER R 0.909/P 1.000。18 FP/79 FN。ローカルで外部送信なし。317 s | **ACCOUNT R 0.755（13 FN）**。PERSON R 0.926（60 FN：カタカナ/1 字/ひらがな）。POSTAL 4 FN（ハイフンなし）、MYNUMBER 1 FN（空白）。gate 差 0.0434。26 dropped。12B の VRAM。`reasoning_effort=none` が必須 | **ローカル 8 ラベルの高精度第一候補**。ACCOUNT/PERSON はルール/モデルで補完 |
| **G ローカル qwen3-swallow** | P 0.9118（B 級）。PERSON/EMAIL/POSTAL/ADDRESS の適合率が高い。MYNUMBER 1.000。2.0 s。ローカル | **R 0.6901 が最低**。PERSON R 0.593、POSTAL R 0.426、ADDRESS R 0.764、PHONE R 0.866、EMAIL R 0.880。**ACCOUNT P 0.296（81 FP）**。506 FN/109 FP。thinking-off の下限。`/no_think` 単独は 17 s/call | **thinking-off では非推奨**。reasoning 有効版は未実行 |

## 8. 手法別の講評

### 8.1 A regex（RuleDetector + GazetteerDetector）

**強み**：形式が閉じた型、チェックディジット、限定辞書で最強である。EMAIL 1.000/1.000、ACCOUNT P 1.000/R 0.981、POSTAL P 0.926/R 0.936、DOB P 1.000、MYNUMBER P 1.000、PHONE P 0.979 を見れば、形式を列挙できる型ではルールで十分だと分かる。0.5 ms/文書、モデルなし、ネットワークなし、完全に決定的で、guard と B/C の土台に適する。

**弱み**：PERSON R 0.116（717 FN）が致命的である。PHONE R 0.835、DOB R 0.817、MYNUMBER R 0.818 にも明確な穴がある。全体 R 0.508 で、単独の PII ツールにはできない。615 個の未注釈予測（DATE 354 / ORG 154 / PATIENT_ID 79 / ROOM 17 / FAX 11）は 8 ラベルでは採点されないが、all-label P を 0.5601、over-mask を 0.4150、IoU を 0.4857 に押し下げる。**これは「8 ラベルの検出ミス」ではなく、corpus が注釈していない型を試みた結果であり、その正しさは未検証**である。表を読む際は分けて考える必要がある。

**結論**：A は「検証器・形式の保険」であり、「氏名検出器」ではない。単独成績はベースラインとしてのみ意味がある。

### 8.2 B CPU モデル単独（sumi-ja-pii int8）

**強み**：**R 0.9945 は全手法で最高**、IoU 0.9424 は A/B/C/D 内で最高（全手法では E 0.9727 / F 0.9640 が上回る）。PERSON R 0.9988、POSTAL/DOB 1.000、PHONE 0.9894。8 ラベルのみを出し、未注釈予測 0、over-mask 0.0039 で境界がきれい。168.4 ms/文書、CPU で動作し、0.995 gate まで 1 TP（9 FN）に迫る。

**弱み**：EMAIL P 0.6529（84 FP）と ACCOUNT P 0.4752（53 FP）は断片化の問題である。1 つのメール/口座が複数 span に分割され、1 対 1 の partial match が余分な断片を FP と数える。PERSON/ADDRESS にも境界 FP が少数ある。**8 ラベルのみ**で、DATE / ORGANIZATION / PATIENT_ID / ROOM / FAX は出さず、ルール層のチェックディジット/構造検証もない。149 FP が主な代償である。

**結論**：B は「再現率優先の神経層」であり、完全なツールではない。ルールと組む際は、EMAIL/ACCOUNT の統合と形式検証をルールが担当し、B が氏名/住所を担当する。19 ラベルの他型は別途補う必要がある。

### 8.3 C regex + CPU（A + B）

**強み**：**F1 0.9796 が最高、P 0.9734**。19 ラベルをカバー。EMAIL/ACCOUNT/MYNUMBER は 1.000、PERSON F1 0.9902、196 ms。B の EMAIL/ACCOUNT 断片化をルールが統合し、適合率を 0.9160 から 0.9734 に上げた。on-prem の単一ツールとして第一候補である。

**弱み**：**R 0.9859 は B の 0.9945 より低い**。原因はモデルではなく、ルール優先の融合である。PHONE 11 件が FAX に、PHONE 7 件が POSTAL_CODE に、DOB 2 件が DATE に、ADDRESS 1 件が ROOM に覆われ、合計 21 / 23 FN を説明する。ADDRESS P 0.9026、POSTAL P 0.9307、PERSON P 0.9818 も、ルール境界による FP を示す。over-mask 0.2732 / IoU 0.6537 は 615 個の未注釈予測の影響で、8 ラベルの検出失敗ではない。

**結論**：C は「成績が最も良い組合せ」だが、fusion に明確な改善余地がある。PHONE/FAX の等価扱い、POSTAL の `〒`/文脈必須化、DOB の DATE 優先、ADDRESS/ROOM の境界統一を行えば、R が B を超える可能性がある。19 ラベルと適合率を維持できる点が強い。

### 8.4 D-v1 ローカル Qwen3-8B（v1 prompt）

**強み**：P 0.9443、F1 0.8470。PERSON R 0.746、ADDRESS R 0.921、EMAIL R 0.943 は A より良い。ローカル 1.14 s/文書、200 calls、19 ラベル許可（176 未注釈）。「ローカル LLM がルールで取れない氏名/住所を補える」ことを示したが、まだ不足している。

**弱み**：POSTAL R 0.2553、DOB R 0.7439、ACCOUNT P 0.5238/R 0.6226、MYNUMBER R 0.8182、PERSON R 0.7460。70 個の `dropped_invalid` は形式遵守の不安定さを示す。v1 prompt は「医療・19 ラベル・残余」向けで、この一般文書/8 ラベル gold には合わない。F1 0.8470 は B 0.9536 / C 0.9796 より低い。

**結論**：D-v1 は歴史的ベースラインであり、「リソースだけ変えても prompt を変えないと不十分」であることを示す。v2 との対照として価値がある。v2 は POSTAL/PHONE/EMAIL/ADDRESS を改善したが、DOB/ACCOUNT が崩れた。

### 8.5 D-v2 ローカル Qwen3-8B（v2 prompt）

**強み**：R 0.8365 は D-v1 の 0.7679 より高い。POSTAL R 0.6915、PHONE R 0.9577、EMAIL R 1.000、ADDRESS R 0.9714、MYNUMBER R 1.000。v2 の 8 ラベル制限で未注釈予測が 176 から 0 になり、`dropped_invalid` も 70 から 13 に減った。prompt の形式・範囲制約は有効である。

**弱み**：**DOB P 0.3913（126 FP）と ACCOUNT P 0.2542（132 FP）** により、全体 precision は D-v1 の 0.9443 から 0.8239 に落ちた。PERSON R も 0.7374。F1 0.8301 は D-v1 の 0.8470 より低い。原因は v2 の方向ではなく、**Qwen3-8B が「生年月日/口座」の文脈制約を守れない**ことにある。すべての日付を DOB、会員/受付番号を ACCOUNT と誤認する。8 ラベルに絞ったため、DATE/ORG/患者 ID なども出さない。

**結論**：D-v2 は「v2 prompt を弱いモデルで使った失敗例」であり、prompt 自体の失敗ではない。E/F が同じ v2 prompt で DOB/ACCOUNT の precision を 0.95/1.00 まで上げていることが証拠である。

### 8.6 E 商用 LLM（DeepSeek-V4.1-Flash-1017 / v2）

**強み**：**P 0.9899 が全手法で最高**、R 0.9559、F1 0.9726 は C に次ぐ。PERSON R 0.9211、PHONE R 0.9965/P 1.000、EMAIL R 0.9810/P 1.000、ADDRESS/POSTAL R 1.000/P 1.000、DOB R 1.000/P 0.9535、ACCOUNT P 1.000/R 0.9245、MYNUMBER R 1.000/P 1.000。FP 16 / FN 72 のみで、8 ラベルの境界は非常にきれい。未注釈 0、`dropped_invalid` 17。D-v2 の DOB/ACCOUNT 崩壊を大幅に修正した。

**弱み**：外部 API であり、費用、DPA、リージョン、保存期間、ベンダー安定性の問題がある。今回 101 件目で 1 回 stall し、停止後に cache から再開したため、レポートは **99 calls + 101 cache** であり、クリーンな 200-call 統計ではない。**信頼できる 200 文書の総時間はない**。PERSON 64 FN が主な穴で、ACCOUNT 4、EMAIL 3（`@sub.example.test`）、PHONE 1。gate 差 0.0391。8 ラベルのみで、C の 19 ラベル代替ではない。

**結論**：E は「外部送信が許される場合の高精度案」である。データを院内から出せない場合は不適。出せるなら、E の precision/F1 は C に近く、ローカル 8B より安定している。

### 8.7 F ローカル Gemma 4 12B QAT（v2 / reasoning_effort=none）

**強み**：**P 0.9885 / R 0.9516 / F1 0.9697** で、E（0.9899/0.9559/0.9726）にほぼ並ぶ。しかも完全にローカルで外部送信がない。PERSON R 0.9260、PHONE R 0.9965/P 0.9930、EMAIL/ADDRESS R 1.000/P 1.000、POSTAL R 0.9574/P 1.000、DOB R 1.000/P 0.9880、ACCOUNT P 1.000/R 0.7547、MYNUMBER R 0.9091/P 1.000。FP 18 / FN 79、317 s / 1,585 ms/文書。

**弱み**：**ACCOUNT R 0.7547（13 FN）が最大の弱点**。PERSON R 0.9260（60 FN：カタカナ/1 字/ひらがな）。POSTAL 4 FN（ハイフンなし 7 桁）、MYNUMBER 1 FN（空白）。gate 差 0.0434。12B の VRAM が必要。`reasoning_effort=none` が必須で、指定しない場合の doc 1 probe は 17.9 s / content 0（max_tokens=1500）。`dropped_invalid` 26。8 ラベルのみ。

**結論**：F は「ローカル 8 ラベルの高精度第一候補」である。ACCOUNT/PERSON をルールや fine-tuned モデルで補えば、F1 はさらに C に近づく。利点はプライバシー（外部送信なし）と精度、代償は 12B のリソースと ACCOUNT 再現率である。

### 8.8 G ローカル qwen3-swallow-8B（v2 / thinking-off）

**強み**：P 0.9118（B 級）。PERSON 0.9959、EMAIL 1.000、POSTAL 1.000、ADDRESS 0.9907、MYNUMBER 0.9167 の適合率が高い。ローカル 2.0 s/文書、200 calls。「出したもの」はほとんど間違えない。

**弱み**：**R 0.6901 が全手法で最低**。PERSON R 0.5931（330 FN）、POSTAL R 0.4255（54 FN）、ADDRESS R 0.7643、PHONE R 0.8662、EMAIL R 0.8797。**ACCOUNT P 0.2957（81 FP）** で、会員/受付番号を口座と誤認する。506 FN / 109 FP、F1 0.7856。`/no_think` 単独は doc 0 で 1.000 だが約 17 s/call。今回の `reasoning_effort=none` は thinking-off の下限であり、**reasoning 有効の 200 件は未実行**。

**結論**：G の thinking-off 版は PII ツールに適さない。reasoning 依存型の可能性が高い。qwen3-swallow の真の能力を評価するには、reasoning 有効版（約 56 分）を別途実行する必要があり、今回の 0.7856 を最終結論にしてはいけない。

## 9. 順位と選定

### 9.1 指標別の順位

| 指標 | 1 位 | 2 位 | 3 位 | 4 位 | 5 位 | 6 位 | 7 位 | 8 位 |
|---|---|---|---|---|---|---|---|---|
| **F1** | **C 0.9796** | E 0.9726 | F 0.9697 | B 0.9536 | D-v1 0.8470 | D-v2 0.8301 | G 0.7856 | A 0.6640 |
| **Recall** | **B 0.9945** | C 0.9859 | E 0.9559 | F 0.9516 | D-v2 0.8365 | D-v1 0.7679 | G 0.6901 | A 0.5083 |
| **Precision** | **E 0.9899** | F 0.9885 | C 0.9734 | A 0.9573 | D-v1 0.9443 | B 0.9160 | G 0.9118 | D-v2 0.8239 |
| **Gate 差（小さい順）** | **B 0.0005** | C 0.0091 | E 0.0391 | F 0.0434 | D-v2 0.1585 | D-v1 0.2271 | G 0.3049 | A 0.4867 |

- **0.995 を通過した手法はない**。B は 1 TP、C は 15 TP、E は 64 TP、F は 71 TP 足りない。
- F1 と Recall の順位は異なる。C は F1 最高だが、Recall は B が最高。E/F は Precision が高いが、Recall は B/C より低い。
- A の Precision 0.9573 は低く見えないが、PERSON Recall 0.116 を見れば「形式に強いだけ」で、PII 検出器ではないと分かる。

### 9.2 用途別の選定

| 用途 | 推奨 | 理由 |
|---|---|---|
| **on-prem、19 ラベル、総合最良** | **C** | F1 0.9796 が最高。19 ラベル。EMAIL/ACCOUNT/MYNUMBER 1.000。196 ms。PHONE/FAX、POSTAL/PHONE、DATE/DOB、ROOM/ADDRESS の fusion 修正が必要 |
| **on-prem、8 ラベル、再現率優先** | **B** | R 0.9945 が最高。IoU 0.9424。168 ms。gate まで 1 TP。EMAIL/ACCOUNT の断片化と 19 ラベルカバレッジをルールで補う |
| **on-prem、8 ラベル、高精度/プライバシー** | **F（Gemma 4 12B）** | P 0.9885 / F1 0.9697。外部送信なし。ACCOUNT R 0.755 が弱点。1,585 ms。12B の VRAM |
| **外部送信可、精度優先** | **E（商用 LLM）** | P 0.9899 が最高。F1 0.9726。8 ラベル。DPA/リージョン/費用が必要。クリーンな 200-call bench なし |
| **形式検証 / guard / 土台** | **A** | EMAIL/ACCOUNT/POSTAL/DOB/MYNUMBER の形式・チェックディジットが強い。0.5 ms。氏名検出には不適 |
| **非推奨** | **G（qwen3-swallow thinking-off）** | R 0.6901、ACCOUNT P 0.296。thinking-off の下限。reasoning 有効版は未実行 |
| **最終案としては非推奨** | **D-v1 / D-v2（Qwen3-8B）** | D-v1 0.8470、D-v2 0.8301。D-v2 は DOB/ACCOUNT の precision が崩壊。E/F に追い越されている |

### 9.3 1 手法だけ選ぶなら

- **on-prem の単一ツール：C**。19 ラベルカバレッジと最高 F1 を両立する唯一の手法。代償は fusion の 23 FN と、615 個の未注釈予測による統計ノイズ。
- **外部の単一ツール：E**。Precision が最高で F1 は C に近い。代償は外部送信と API 安定性。
- **ローカル 8 ラベル：F**。Precision は E に近く、外部送信がない。ACCOUNT/PERSON の補完が必要。
- **再現率優先：B**。gate に最も近いが、EMAIL/ACCOUNT の断片化と 8 ラベル範囲が制約。

### 9.4 主な改善方向

| 手法 | 最優先の修正 | 期待効果 |
|---|---|---|
| C | fusion の仲裁：PHONE/FAX の等価扱い、POSTAL の `〒`/文脈必須化、DOB の DATE 優先、ADDRESS/ROOM の統一 | R 0.9859 → B を超える可能性。19 ラベルと Precision を維持 |
| B | EMAIL/ACCOUNT の断片統合、DATE/ORG/患者 ID/ROOM/FAX の追加 | F1 とカバレッジの向上。ルール層が必要 |
| F | ACCOUNT の再現率（13 FN）と PERSON のカタカナ/1 字/ひらがな（60 FN） | F1 が C に接近。ローカル/外部送信なしを維持 |
| E | PERSON 64 FN、API timeout/再開戦略、費用/DPA | gate に接近。本番安定性 |
| A | PERSON。氏名は B/C/F に任せる | guard 専用として割り切る |
| G | reasoning 有効で再評価、ACCOUNT の負例/文脈 | 真の能力を判断（約 56 分） |
| D-v2 | DOB/ACCOUNT の負例/文脈。ただし E/F に追い越されている | prompt 反復のベースラインのみ |

## 10. 制限と誤読しやすい点

1. **同源データ**: `sumi-ja-test` は `sumi-ja-pii` の学習データと同源である。B/D-v2/E/F/G の高スコアは**分布内の上界**であり、臨床精度ではない。A/C も同じデータの制約を受ける。
2. **一般文書**: sumi は議事録、請求書、申込書、法令、青空文庫などで、progress note ではない。SOAP、医療略号、検査値表、紹介状、OCR 崩れは未検証である。
3. **gold があるのは 8 ラベルのみ**: DATE / ORGANIZATION / PATIENT_ID / ROOM / FAX には gold がない。A/C の 615 個、D-v1 の 176 個の未注釈予測は、**誤りとも正解とも言えず**、8 ラベルの P/R には入らない。
4. **over-mask / IoU / all-label P は直接横比較できない**: A/C の 615 個の未注釈予測が over-mask を上げ、IoU/all-label P を下げる。B/D-v2/E/F/G の over-mask が低いのは、**8 ラベルしか出さない**ためであり、「より安全」という意味ではない。A の all-label P 0.5601 は「8 ラベルの precision」ではなく、8 ラベルの precision は 0.9573 である。
5. **速度条件は完全には同じでない**: A/B/C/D-v1/D-v2 は `bench`、F/G は eval の wall time、E は再開実行の参考値。A の 0.5 ms と E の「約 1.0 s/call」を単純に比較してはいけない。
6. **E にはクリーンな 200-call bench がない**: 初回 101 件で 1 回 API が stall し、停止後に cache から再開した。レポートの 100 秒は 99 calls + 101 cache hits の再開時間で、200 calls の総時間ではない。E の `calls=99 / cache=101 / chars=45,515` は**部分統計**である。
7. **G は thinking-off の下限**: `/no_think` 単独は 1 件で 1.000 だが約 17 s/call。今回の `reasoning_effort=none` による 200 件は 0.6901 である。**reasoning 有効版は未実行**で、0.7856 を qwen3-swallow の最終能力と見てはいけない。
8. **F には `reasoning_effort=none` が必要**: 指定しない場合の doc 1 probe は 17.9 s / content 0（max_tokens=1500）。これはモデル固有の挙動で、prompt/8 ラベル/評価は同じだが生成制御が異なる。
9. **D-v1 の v1 prompt は 19 ラベルの全量抽出版ではない**: 医療・19 ラベル・残余向けであり、v2 の一般・8 ラベルとは同一 prompt ではない。D-v1 と D-v2 の差には prompt 変更が含まれ、すべてをモデル差に帰属できない。
10. **gate は点推定**: 0.995 はプロジェクトの受入ゲート。B 0.9945（1 TP 不足）、C 0.9859、E 0.9559、F 0.9516 は未通過。CI 下限も 0.995 未満（B 0.9919、C 0.9790、E 0.9353、F 0.9268）。「近い」を「通過」と書いてはいけない。
11. **検出のみを評価**: surrogate/placeholder、日付の平行移動、guard、下流 NER/ICD の可用性、人手レビューコストは評価していない。
12. **実患者データはない**: すべて 200 文書の公開合成/一般文書に基づく。実カルテでは再注釈・再評価が必要である。

## 11. 最終講評

### 11.1 総合判断

**C は on-prem 単一ツールの現時点の最良解、E は外部送信時の高精度解、F はローカル 8 ラベルの高精度解、B は再現率最高の神経層、A は形式/guard の土台、G の thinking-off は不適、D-v1/D-v2 は prompt/モデル反復の歴史的ベースラインである。**

この順位は「スコアが高い＝そのまま本番に適する」ではない。目的の違いによる折衷である。

- **19 ラベルを 1 ツールで**: C（F1 0.9796、19 ラベル、196 ms）。
- **8 ラベルで再現率最優先**: B（R 0.9945、IoU 0.9424、168 ms）。
- **8 ラベルでローカル高精度/プライバシー**: F（P 0.9885 / F1 0.9697、1,585 ms）。
- **8 ラベルで外部送信可・最高精度**: E（P 0.9899 / F1 0.9726）。
- **形式検証/guard**: A（EMAIL/ACCOUNT/DOB/MYNUMBER の precision 1.000、0.5 ms）。
- **使わない**: G の thinking-off（R 0.6901、ACCOUNT P 0.296）。D-v2 の Qwen3-8B 版（DOB P 0.391、ACCOUNT P 0.254）。

### 11.2 各手法の実際の位置づけ

**A regex**：非常に良い「形式検証器」だが、「氏名検出器」ではない。EMAIL、ACCOUNT、DOB、MYNUMBER の precision は 1.0、POSTAL は 0.926 で、閉じた文法ではルールが不可欠である。一方、PERSON 0.116 は単独で PII 検出を担えないことを示す。guard、チェックディジット、形式の保険、B/C/F への構造化ルール提供が正しい役割である。

**B CPU モデル**：再現率最強の 8 ラベル神経層である。PERSON 0.999、POSTAL/DOB 1.000、gate まで 1 TP。ただし EMAIL/ACCOUNT を断片化し（P 0.653 / 0.475）、8 ラベルしか出さない。完全なツールではないが、C の主要な再現率源である。

**C regex+CPU**：現時点の最良の単一ツールだが、再現率は B より低い。原因はルール優先の融合が正しいモデル span を覆うこと（PHONE/FAX 11、POSTAL/PHONE 7、DATE/DOB 2、ROOM/ADDRESS 1）。**C の弱点はモデルではなく仲裁ロジック**である。PHONE/FAX の等価扱い、POSTAL の `〒`/文脈必須化、DOB の DATE 優先、ADDRESS/ROOM の統一で、B を超える可能性がある。19 ラベルと precision を維持できる点が強い。

**D-v1 / D-v2**：prompt とモデル能力の一致が必要であることを示す。D-v1 の v1 prompt は医療/19 ラベル/残余向けで、POSTAL 0.255、DOB 0.744。D-v2 の v2 prompt は POSTAL 0.692、PHONE 0.958、EMAIL 1.000 まで改善したが、Qwen3-8B が DOB precision 0.391、ACCOUNT 0.254 に落とした。**v2 prompt の失敗ではなく、8B モデルが DOB/ACCOUNT の文脈を守れない**。E/F が同じ v2 prompt で機能することが証拠である。

**E 商用 LLM**：精度最高の手法で、P 0.9899、F1 0.9726。D-v2 の DOB/ACCOUNT 崩壊を大幅に修正した。代償は外部送信、費用、DPA、リージョン、ベンダー安定性、1 回の stall による非クリーンな 200-call 統計である。「院内から出せる、precision 最優先」の場面に向く。

**F ローカル Gemma 4 12B**：ローカル 8 ラベルの高精度最有力で、P 0.9885 / F1 0.9697。E にほぼ並び、外部送信がない。主な課題は ACCOUNT R 0.755（13 FN）、PERSON 0.926（60 FN：カタカナ/1 字/ひらがな）、12B の VRAM、1,585 ms。`reasoning_effort=none` が必須で、指定しない場合の doc 1 probe は 17.9 s / content 0（max_tokens=1500）。「ローカル、8 ラベル、精度優先」に向く。

**G qwen3-swallow**：thinking-off の負の結果である。P 0.9118 は低く見えないが、R 0.6901、ACCOUNT P 0.296 で、出力が少なく、会員/受付番号を口座と誤認する。`/no_think` 単独は 1 件 1.000 だが約 17 s/call。今回の `reasoning_effort=none` は下限であり、**0.7856 をモデルの真の能力と判断してはいけない**。判断するには reasoning 有効版（約 56 分）が必要である。

### 11.3 「漏えい 0」を目指す場合

0.995 を通過した手法はない。B 0.9945 が最も近く（1 TP 不足）、C 0.9859（15 TP 不足）、E 0.9559（64 TP 不足）、F 0.9516（71 TP 不足）である。したがって、

1. **どの単独手法も「漏えい 0」の保証にはならない**。
2. on-prem では **C** を土台にし、fusion を修正し、guard と人手レビューで補う。
3. 8 ラベルの高精度と 12B ローカル資源を許容できるなら、**F + ルール（ACCOUNT/PERSON）** が D-v2 より現実的である。
4. 外部送信が可能なら **E** の precision が最も高いが、PERSON 64 FN が残るため、guard/人手レビューは必要である。
5. 最終受入は実カルテ 200–500 文書の二重注釈で行う。この同源 200 文書で代用してはいけない。

### 11.4 一言でまとめる

**C は on-prem の総合最良、E は外部の精度最良、F はローカル 8 ラベルの最良、B は再現率最良、A は形式/guard 最良である。G の thinking-off は不適、D-v1/D-v2 は歴史的ベースライン。すべてのスコアは同源の一般文書 200 文書による分布内の上界であり、臨床精度ではない。0.995 ゲートを通過した手法はない。**

---

## 12. 参考: phi-deidentification-bert（外部ツール、200 文書 smoke 評価）

> **この節は A/B/C/D/E/F/G の 200 文書ランキングに追加するものではない。**
> 外部 BERT リポジトリは公式の学習済みモデル重みを公開していないため、以下は
> **jmtt1-5k の 797 篇で学習した smoke checkpoint** を `sumi-ja-test` 全 200 文書に適用した
> 参考結果である。公式 BERT モデルの成績ではない。

### 12.1 前提: 公式モデルの重みは未公開

- リポジトリ: `external/phi-deidentification-bert/`（`sociocom/phi-deidentification-bert`、commit `d166787`）
- GitHub に release / tag / LFS のモデル重みなし
- HuggingFace に公開 `sociocom/phi-deidentification-bert` checkpoint なし（直接アクセスは 401）
- リポジトリに `.data` の学習データなし

そのため、今回の BERT は `sbintuitions/modernbert-ja-310m` を
jmtt1-5k の 797 篇で 1 epoch fine-tune した **smoke checkpoint**
（`external/phi-deidentification-bert/out_jmtt_smoke/`）である。
公式モデルでの 200 文書評価は、公式 checkpoint が提供されるまで不可能である。

### 12.2 ラベル対応（重要）

BERT のラベルは sumi の 8 ラベルと 1 対 1 に対応しない。
以下の 3 つの見方を分けて評価した。

| BERT ラベル | sumi への対応 | 備考 |
|---|---|---|
| `phi_person` | PERSON | 1 対 1 |
| `phi_tel` | PHONE | 1 対 1。ただし smoke 学習データに電話例がなく、出力 0 |
| `phi_location` | ADDRESS + POSTAL_CODE | 1 対多（duplicate） |
| `phi_id` | EMAIL + ACCOUNT + MYNUMBER | 1 対多（duplicate） |
| `phi_age` | DOB | 学習時に DOB を `phi_age` に写像したため |
| `phi_hospital` | ORGANIZATION | sumi の 8 ラベル外（採点外） |
| `phi_job` | JOB | sumi の 8 ラベル外（採点外） |

- **strict**: 1 対 1 で明確な PERSON / PHONE / ADDRESS / EMAIL のみ。
- **expanded**: 上表の 1 対多を duplicate して 8 ラベルを無理に埋める best-effort。
- **agnostic**: ラベルを無視し、任意の PHI span が任意の gold span と重なるかを見る。

### 12.3 200 文書の全体結果

| 見方 | 対象ラベル | 予測数 | P | R | F1 | critical R | over-mask | char IoU | support |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict | PERSON / PHONE / ADDRESS / EMAIL | 1,633 | **0.6246** | **0.7322** | **0.6742** | 0.7322 | 0.0949 | 0.5795 | 1,393 |
| expanded | 8 ラベル（duplicate あり） | 3,459 | 0.3814 | 0.6840 | 0.4897 | 0.6840 | 0.0882 | 0.5170 | 1,633 |
| agnostic | ラベル無視（any PHI） | 3,459 | 0.4111 | **0.8708** | 0.5585 | 0.8708 | 0.0882 | 0.5170 | 1,633 |

- **strict 4 ラベル**: P 0.6246 / R 0.7322 / F1 0.6742。
  これは 8 ラベル全体の成績ではない。A の 8 ラベル F1 0.6640 と直接比較してはいけない。
- **expanded 8 ラベル**: P 0.3814 / R 0.6840 / F1 0.4897。
  `phi_location` を ADDRESS/POSTAL_CODE、`phi_id` を EMAIL/ACCOUNT/MYNUMBER に duplicate しているため、
  FP が大幅に増える。8 ラベルの公式成績ではない。
- **agnostic**: R 0.8708 / P 0.4111 / F1 0.5585。
  ラベルを無視すると gold の 87.1% を拾えるが、予測の 58.9% はどの gold とも重ならない。
  「検出能力」の参考値であり、8 ラベルの P/R/F1 ではない。


### 12.4 ラベル別結果（expanded 8 ラベル）

`phi_location` を ADDRESS/POSTAL_CODE、`phi_id` を EMAIL/ACCOUNT/MYNUMBER に
duplicate した best-effort の結果。duplicate により FP が大きく出る。

| label | support | pred | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| PERSON | 811 | 755 | **0.956** | **0.890** | **0.922** |
| ADDRESS | 140 | 465 | 0.301 | **1.000** | 0.463 |
| EMAIL | 158 | 413 | 0.383 | **1.000** | 0.553 |
| POSTAL_CODE | 94 | 465 | 0.202 | **1.000** | 0.336 |
| PHONE | 284 | 0 | 0.000 | 0.000 | 0.000 |
| DOB | 82 | 5 | 0.200 | 0.012 | 0.023 |
| ACCOUNT | 53 | 413 | 0.005 | 0.038 | 0.009 |
| MYNUMBER | 11 | 413 | 0.000 | 0.000 | 0.000 |
| ORGANIZATION | 0 | 530 | — | — | — |

- **PERSON だけが実用域**: P 0.956 / R 0.890 / F1 0.922。
- **PHONE は 0/284**: smoke 学習データ（jmtt1-5k）に電話番号がなく、
  `phi_tel` を学習できていない。公式 BERT の能力とは切り離して読む必要がある。
- **EMAIL / ADDRESS / POSTAL_CODE は recall 1.000、precision 0.20〜0.38**:
  `phi_location` / `phi_id` が広く出すぎており、duplicate も重なって FP が増えている。
- **DOB は 5 件しか出ない**: jmtt の DOB は 19 件だけであり、`phi_age` への写像もあって
  ほとんど学習できていない。
- **ACCOUNT / MYNUMBER は実質 0**: `phi_id` の duplicate だけでは足りない。
- **ORGANIZATION 530 件は sumi の 8 ラベル外**なので採点外。FP とも FN とも言えない。

### 12.5 講評

- **PERSON は cross-domain で強い**。jmtt で学習した smoke checkpoint が
  sumi の人名 811 件に対して F1 0.922 を出した。
- **PHONE/EMAIL の境界は弱い**。PHONE は学習例なしで 0、EMAIL は recall 1.000 だが
  precision 0.383。形式が閉じた PHONE/EMAIL はルール層で補うべきである。
- **ADDRESS/POSTAL_CODE は recall 優先で over-predict**。`phi_location` の広い検出を
  duplicate しているため、precision 0.20〜0.30 は「モデルの実力」というより
  マッピングの副作用が大きい。
- **DOB/ACCOUNT/MYNUMBER はこの smoke checkpoint では評価不能**。
  学習データの不足とラベル体系の不一致が重なっている。
- **agnostic では recall 0.871 / precision 0.411**。
  ラベルを無視すれば gold の 87% を拾えるが、予測の約 59% は gold と重ならない。
  「検出能力はあるが、境界とラベルが粗い」と読むのが妥当。
- **200 文書の offset 整合性**: tagged/plain の alignment mismatch は **0 件**。
  ただし smoke checkpoint であり、公式 BERT の 200 文書成績ではない。


### 12.6 既存手法との並置（順位ではない）

| 手法 | データ | ラベル | P | R | F1 | 備考 |
|---|---|---|---:|---:|---:|---|
| A regex | sumi 200 | 8 | 0.9573 | 0.5083 | 0.6640 | 既存 |
| B CPU モデル | sumi 200 | 8 | 0.9160 | 0.9945 | 0.9536 | 既存 |
| C regex+CPU | sumi 200 | 8 | 0.9734 | 0.9859 | 0.9796 | 既存 |
| E 商用 LLM | sumi 200 | 8 | 0.9899 | 0.9559 | 0.9726 | 既存 |
| F Gemma 4 12B | sumi 200 | 8 | 0.9885 | 0.9516 | 0.9697 | 既存 |
| G qwen3-swallow | sumi 200 | 8 | 0.9118 | 0.6901 | 0.7856 | 既存 |
| **BERT strict** | sumi 200 | **4** | 0.6246 | 0.7322 | 0.6742 | smoke、4 ラベルのみ |
| **BERT expanded** | sumi 200 | **8（duplicate）** | 0.3814 | 0.6840 | 0.4897 | smoke、duplicate で FP 増 |
| **BERT agnostic** | sumi 200 | **any** | 0.4111 | 0.8708 | 0.5585 | smoke、ラベル無視 |

- **BERT strict の 0.6742 を A の 0.6640 と同列に置かない**。BERT strict は
  PERSON/PHONE/ADDRESS/EMAIL の 4 ラベルだけであり、A は 8 ラベルである。
- **BERT expanded の 0.4897 は 8 ラベルだが、duplicate による FP を含む**。
  公式 BERT の 8 ラベル成績ではない。
- **BERT agnostic の 0.5585 はラベルを無視した検出能力**であり、
  既存手法の same-label F1 と直接比較できない。
- したがって、**この BERT 結果を 2.1 のランキングに挿入しない**。

### 12.7 第一篇 `sumi-test-00000` の例

- gold: PERSON 6、PHONE 1、EMAIL 1
- BERT の strict 確認:
  - PERSON **6/6 完全一致**
  - PHONE **0/1**（`054-908-9723` が `0` / `0` / `7` などに断片化）
  - EMAIL **0/1**（`itsuki.yamashita@` と `jp` に断片化）

出力例:

```text
出席者 <phi_person>フクシママイ</phi_person>(品質保証部)、
<phi_person>石井 薫</phi_person>(カスタマーサポート部)、
<phi_person>山下樹</phi_person>(<phi_hospital>青葉システムズ株式会社</phi_hospital>)
...
問い合わせ窓口は <phi_hospital>0</phi_hospital>54-9<phi_location>0</phi_location>8-9<phi_location>7</phi_location>23
...
資料送付先は<phi_id> itsuki.yamashita@</phi_id>example.or.<phi_id>jp</phi_id>
```

参考: `jmtt1-5k` の第一篇 `4p欠失症候群` では、
`MedicalNote, Inc.` が `<phi_hospital>MedicalNote, Inc.</phi_hospital>` として exact match。

### 12.8 関連ファイル

- 全 200 文書の BERT 入力: `external/phi-deidentification-bert/.data/sumi_test_200.txt`
- 全 200 文書の BERT 出力: `external/phi-deidentification-bert/.data/sumi_test_200_result.txt`
- 全 200 文書の評価 JSON: `reports/20261119-phi-bert-sumi.json`
- BERT 推論ログ: `out/phi_bert_infer_sumi_200.log`
- BERT 評価ログ: `out/phi_bert_eval_sumi_200.log`
- 評価スクリプト: `scripts/eval_phi_bert_sumi.py`
- 統合メモ: `docs/phi-deidentification-bert-integration.md`

### 12.9 結論

- **200 文書の全量実行は完了**した。offset 整合性は 0 問題。
- ただし、これは **公式 BERT ではなく smoke checkpoint** であり、
  ラベル体系も sumi の 8 ラベルと 1 対 1 ではない。
- **PERSON は F1 0.922 で強い**が、**PHONE 0/284、DOB/ACCOUNT/MYNUMBER はほぼ 0**。
- **公式 BERT の 200 文書成績として引用してはいけない**。
- 公式 checkpoint が提供されたら、同じ `inference.py` / `scripts/eval_phi_bert_sumi.py` で
  200 文書を再評価する。そのとき初めて A/B/C/E/F/G と同じ表に置ける。

## 13. 追加: MedPHINER-Llama-3.1-Swallow-8B（native `<phi_*>` prompt）

> **この節は A–G のランキングに追加するものではない。**
> MedPHINER は公式の `sociocom/MedPHINER-Llama-3.1-Swallow-8B` を使ったが、
> Gemma 4 の JSON prompt を守れないため、**native `<phi_*>` prompt** で評価した。
> 両データセットで同じ prompt / 同じ設定を使っている。

### 13.1 設定

| 項目 | 内容 |
|---|---|
| モデル | `medphiner-llama-3.1-swallow-8b-instruct-v0.5` |
| prompt | `medphiner-native-v1`（`<phi_*>` タグ、7 ラベル） |
| 生成 | `temperature=0.0`, `max_tokens=8192`, 1 call/doc, `max_chars=4000` |
| 評価 | 200 文書、4 workers |
| 入力 | `data/public/sumi-ja-test.jsonl` |
| 出力 | `out/medphiner-sumi-full/sumi_spans.jsonl` |
| レポート | `reports/20261119-medphiner-sumi.json` |

- 200/200 完成、0 失敗。
- `plain_match=True` は 146/200。**54 篇はモデルが本文を途中で止めたため incomplete**。
  そのため recall は**下限**である。
- `phi_email` は EMAIL に、`phi_id` は ACCOUNT/MYNUMBER に、`phi_location` は
  ADDRESS/POSTAL_CODE に、`phi_age` は DOB に写像した。`phi_job` は破棄。

### 13.2 全体結果

| 見方 | 対象ラベル | 予測数 | P | R | F1 | critical R | over-mask | char IoU | support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict | PERSON / PHONE / ADDRESS / EMAIL | 1,234 | **0.9165** | **0.8119** | **0.8611** | 0.8119 | 0.0616 | 0.7211 | 1,393 |
| expanded | 8 ラベル（duplicate） | 2,080 | 0.6042 | 0.7244 | 0.6589 | 0.7244 | 0.3639 | 0.5221 | 1,633 |
| agnostic | ラベル無視（any PHI） | 2,080 | 0.5793 | **0.7379** | 0.6491 | 0.7379 | 0.3639 | 0.5221 | 1,633 |

### 13.3 ラベル別結果（expanded）

| label | support | pred | P | R | F1 |
|---:|---:|---:|---:|---:|---:|
| PERSON | 811 | 775 | **0.932** | **0.890** | **0.910** |
| PHONE | 284 | 247 | **0.931** | 0.810 | 0.866 |
| EMAIL | 158 | 128 | **1.000** | 0.810 | 0.895 |
| ADDRESS | 140 | 84 | 0.607 | 0.364 | 0.455 |
| POSTAL_CODE | 94 | 84 | 0.059 | 0.053 | 0.056 |
| ACCOUNT | 53 | 319 | 0.110 | 0.660 | 0.188 |
| MYNUMBER | 11 | 319 | 0.035 | 1.000 | 0.067 |
| DOB | 82 | 2 | 0.500 | 0.012 | 0.024 |
| ORGANIZATION | 0 | 122 | — | — | — |

- PERSON / PHONE / EMAIL は強い。特に EMAIL P 1.000。
- **ADDRESS / POSTAL_CODE は弱い**。`phi_location` が住所全体を拾えず、
  duplicate で POSTAL の FP も増える。
- **ACCOUNT / MYNUMBER は precision が低い**。`phi_id` が口座・会員番号・
  受付番号などを一括してしまう。
- **DOB はほぼ 0**。`phi_age` は年齢に寄り、生年月日を拾えない。
- ORGANIZATION 122 は sumi の 8 ラベル外なので採点外。

### 13.4 講評

- strict 4 ラベル F1 **0.8611** は高いが、PERSON/PHONE/ADDRESS/EMAAL の 4 ラベルだけ。
- expanded 8 ラベル F1 **0.6589**。A regex 0.6640 と同程度だが、
  `over-mask 0.3639`、`char IoU 0.5221` で境界・過剰検出は悪い。
- **54/200 篇が incomplete** なので、本来の recall はこれより高い可能性がある。
  一方、incomplete を補うには chunk 分割が必要で、それは「1 call/doc」条件を崩す。
- MedPHINER は **native prompt でないと動かない**。Gemma 4 の JSON prompt では
  0 件になるため、**Gemma 4 と同じ prompt での比較は不能**である。
- I は A–G のランキングに混ぜない。詳細は `20261119-jmtt1-5k-regex-cpu-result-ja.md` 第 13 節。

### 13.5 関連ファイル

- `reports/20261119-medphiner-sumi.json`
- `out/medphiner-sumi-full/sumi_spans.jsonl`
- `out/medphiner_sumi_full.log`
- `out/medphiner_eval_sumi.log`
- `scripts/run_medphiner.py`
- `scripts/eval_phi_bert_sumi.py`


