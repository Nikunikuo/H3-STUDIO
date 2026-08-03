# Model Terms / モデル利用条件

最終確認: 2026-08-04

H3-STUDIO本体のソースコードには、ルートの [`LICENSE`](./LICENSE) にあるMIT Licenseが適用されます。MiniMax H3のモデル、重み、派生重みにはMIT Licenseは適用されません。

This repository's source code is MIT-licensed. MiniMax H3 models and weights are governed by their own terms.

## 重みは同梱しません / Weights are not bundled

Gitリポジトリにはモデル重みを含めません。初回セットアップ時に、Comfy-OrgがMiniMax H3向けに公開した5ファイル（合計63,440,965,087 bytes、約59.08 GiB）を固定revision `0543966fbdce5ba05709a8f2031c94bdba629b4a`から取得し、サイズとSHA-256を [`comfy_models.lock.json`](./comfy_models.lock.json) に照合します。元となるMiniMax H3の利用条件はMiniMax公式ライセンスに従います。

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

## セットアップ時の確認 / Setup acknowledgement

`Setup-H3-Studio.cmd`は、巨大ファイルを取得する前にモデルライセンスを表示して明示確認を求めます。PowerShellからモデルを取得するセットアップを直接実行する場合は、内容を確認したうえで `scripts\setup_comfy.ps1 -AcceptMiniMaxH3License` を指定します。既に全モデルが正しいサイズで存在する検査・修復では、再ダウンロードがないためこのスイッチを要求しません。このスイッチは確認記録とダウンロード制御のためのものであり、ライセンス上の権利を追加したり制限を解除したりするものではありません。

モデルを利用、複製、変更、配布、実行または表示する行為自体が、公式ライセンス上の同意行為になり得ます。利用する時点・地域・用途について、必ず公式原文を確認してください。

## Official redistribution notice

モデルライセンスがMiniMax H3 Worksの再配布時に指定している通知文は次のとおりです（H3-STUDIOのGitリポジトリ自体は重みを配布しません）。

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.
