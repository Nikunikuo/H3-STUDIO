# Model Terms / モデル利用条件

最終確認: 2026-08-05

H3-STUDIO本体のソースコードには、ルートの [`LICENSE`](./LICENSE) にあるMIT Licenseが適用されます。MiniMax H3、必須のQwenプロンプトプランナー、および任意の旧LFM翻訳モデルのモデル、重み、派生重みにはMIT Licenseは適用されません。

This repository's source code is MIT-licensed. MiniMax H3, Qwen prompt-planner, and optional LFM prompt-translator models and weights are governed by their own terms.

## 重みは同梱しません / Weights are not bundled

Gitリポジトリにはモデル重みを含めません。初回セットアップ時に、Comfy-OrgがMiniMax H3向けに公開した5ファイル（合計63,440,965,087 bytes、約59.08 GiB）を固定revision `0543966fbdce5ba05709a8f2031c94bdba629b4a`から取得し、サイズとSHA-256を [`comfy_models.lock.json`](./comfy_models.lock.json) に照合します。元となるMiniMax H3の利用条件はMiniMax公式ライセンスに従います。

既定のcommunity prompt plannerには`Qwen/Qwen3-4B-Instruct-2507`を使用します。セットアップは、固定revision `cdbee75f17c01a7cc42f958dc650907174af0554`から実行に必要な9ファイル（合計8,056,459,158 bytes、約7.50 GiB）だけを`models/prompt_planner/Qwen3-4B-Instruct-2507`へ取得し、[`prompt_planner.lock.json`](./prompt_planner.lock.json)のサイズとSHA-256へ照合します。確認済み上流snapshot全体は8,060,917,568 bytesですが、README、`.gitattributes`、高速Tokenizerと重複するslow-tokenizer用語彙ファイルは配布導線の許可リストから除外します。Qwenの重みはGitに含めません。

任意の旧A/B比較でLiquid AIの`LFM2-350M-ENJP-MT`を使用する場合だけ、セットアップで導入を選びます。8ファイル（合計713,822,225 bytes、約680.75 MiB）を固定revision `80367784d525777ad7565b24534ba5810eeac59f`から`models/prompt_translator`へ取得し、[`prompt_translator.lock.json`](./prompt_translator.lock.json)に記録したサイズとSHA-256へ照合します。この重みもGitには含めません。LFMを省略しても、既定のQwen community plannerと英語promptのadvanced pass-throughは利用できます。

## 正式な利用条件 / Controlling terms

- 固定した初回公開revisionのライセンス: [MiniMax H3 Community License Agreement (immutable revision)](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/af0fe5abe6fd50d632b65a82fef321c4c5c1f249/LICENSE)
- 現行の公式ライセンス: [MiniMax H3 LICENSE (current)](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- 公式モデルカード: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- 公式ライセンスQ&A: [Q&A About License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)

以下は重要項目の非網羅的な要約であり、法的助言ではありません。原文と異なる場合は公式ライセンスが優先します。

- 適用地域は、EU、英国、韓国、米国を除く全世界です。ライセンスは、モデルやその出力を適用地域外で利用・複製・変更・配布・表示することを許可していません。
- 法令およびAcceptable Use Policyへの準拠が必要です。危害、未成年者の搾取、権利侵害、選挙へ影響させる虚偽情報、マルウェア、無断のなりすまし、高リスク領域の自動意思決定、軍事目的など、原文に列挙された用途は禁止されています。公開環境へ生成物を出す場合のAI生成表示要件もあります。
- MiniMax H3またはそのModel Derivatives以外のAIモデルを改善する目的で、MiniMax H3 Worksまたはその出力・結果を使用できません。
- 重みやModel Derivativesを再配布する場合は、受領者へのライセンス提供、変更ファイルへの表示、所定の`NOTICE`、地域制限を含む再配布条件を満たす必要があります。
- MiniMax H3を組み込んだ製品・サービスを第三者へ提供する場合は、利用者を所定の制限へ拘束し、合理的な安全対策と違反報告手段を維持する必要があります。
- 年間売上が2,000万米ドルを超える商用製品・サービスにはMiniMaxの事前書面承認が必要です。商用UIには`MiniMax H3`の表示要件があります。

## 必須プロンプトプランナー / Required prompt planner

- 固定revisionのモデルカード: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/README.md)
- 固定revisionのライセンス: [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE)
- 現行ライセンス: [Qwen3-4B-Instruct-2507 LICENSE (current)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE)

`Qwen3-4B-Instruct-2507`はApache License 2.0で公開されています。セットアップに追加のクリック同意は設けませんが、著作権、ライセンス、NOTICE、特許、商標等の条件は公式原文が支配します。H3 Studioは重みを再配布せず、固定revisionから直接取得して上流`LICENSE`をモデルディレクトリへ保存します。モデルは日本語自然文の意味展開だけを別processで行い、コード側が参照番号、台詞原文、audio policy、strict JSON schemaと公開成功例型の英語ブロック構造を検証します。コンパイルprocessは完了後に終了し、H3／ComfyUIと同時常駐しません。

## 任意の旧プロンプト翻訳モデル / Optional legacy prompt translator

- 固定revisionのモデルカード: [LiquidAI/LFM2-350M-ENJP-MT](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/blob/80367784d525777ad7565b24534ba5810eeac59f/README.md)
- 固定revisionのライセンス: [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/blob/80367784d525777ad7565b24534ba5810eeac59f/LICENSE)
- 現行ライセンス: [LFM2-350M-ENJP-MT LICENSE (current)](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/blob/main/LICENSE)

次はLFM Open License v1.0の非網羅的な要約であり、法的助言ではありません。必ず上記の公式原文を確認してください。

- 利用、複製、派生物の作成、表示、再許諾、配布に関する権利は、ライセンスに定める条件と商用利用制限に従うことを条件として付与されます。
- 商用利用は年間売上1,000万米ドルの基準によって制限されます。ライセンスSection 5は、その基準を超えるLegal Entityによる商用利用をこの契約では許諾していません。該当性は公式原文で判断してください。
- 重みまたは派生物を再配布する場合は、受領者へのライセンス写しの提供、変更ファイルへの明示、著作権・特許・商標・帰属表示の保持、上流に`NOTICE`がある場合の通知保持など、Section 4の条件を満たす必要があります。
- 商標の利用許可は、由来を説明する合理的な通常利用など、ライセンスに明記された範囲に限定されます。
- 本GitリポジトリはLFM2の重みを再配布しません。セットアップが固定revisionから直接取得し、上流の`LICENSE`もモデルディレクトリ内へ保存します。
- この翻訳モデルは任意の旧比較コンポーネントです。導入しない選択はMiniMax H3本体、既定のQwen community planner、英語入力、既に正しい公式形式の入力を制限しません。旧LFMとのA/B比較が必要になった時点で、別途ライセンスを確認して導入できます。

## セットアップ時の確認 / Setup acknowledgement

`Setup-H3-Studio.cmd`は、ファイルを取得する前にMiniMax H3の公式ライセンスを表示し、明示確認を必須にします。Apache-2.0のQwen plannerは追加同意なしで必須コンポーネントとして取得します。通常UIで使用しない旧LFM翻訳モデルは既定で省略されます。必要な上級者だけ、LFM Open License v1.0を別途確認したうえで`-AcceptPromptTranslatorLicense`を明示して導入します。

PowerShellから通常セットアップを直接実行すると、Qwen plannerは固定revisionと許可リストで自動取得されます。初回セットアップは9ファイルのサイズ／SHA-256照合、offline設定／Tokenizer検査、CPUでの全重みロードprobeまで行い、全件一致後にH3 Studio管理の来歴markerを原子的に作成します。通常起動の高速`-VerifyOnly -SkipModelHash`は全ファイルのサイズ、offline設定／Tokenizer互換、markerのモデルID・revision・lock SHA-256・件数・総容量を検査し、必須Qwenが欠けているか来歴が一致しなければ起動せず再セットアップを案内します。

任意の旧LFMが未導入または正常な途中ダウンロード状態なら、導入には`-AcceptPromptTranslatorLicense`、導入しない場合は`-SkipPromptTranslator`を指定します。LFMがなくても`-VerifyOnly`と通常の`Start-H3-WebUI.cmd`は成功し、Qwen community plannerを利用できます。一部だけ存在するLFMや破損状態は曖昧に利用せず停止し、修復するか`-SkipPromptTranslator`を明示する必要があります。これらのスイッチは確認、ダウンロード、機能選択のためのものであり、ライセンス上の権利を追加したり制限を解除したりするものではありません。

モデルを利用、複製、変更、配布、実行または表示する行為自体が、公式ライセンス上の同意行為になり得ます。利用する時点・地域・用途について、必ず公式原文を確認してください。

## Official redistribution notice

モデルライセンスがMiniMax H3 Worksの再配布時に指定している通知文は次のとおりです（H3-STUDIOのGitリポジトリ自体は重みを配布しません）。

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

LFM2の重みまたは派生物を別途再配布する場合は、固定revisionに同梱されたLFM Open License v1.0のSection 4を確認し、ライセンス写し、変更表示、帰属表示および該当する`NOTICE`を保持してください。
