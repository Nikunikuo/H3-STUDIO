# Third-Party Notices / サードパーティ通知

最終確認: 2026-08-04

H3-STUDIO本体のMIT Licenseは、セットアップ時に取得される上流ソフトウェア、Python package、モデル、重みを再ライセンスするものではありません。それぞれの著作権表示とライセンスが維持されます。モデル固有の条件は [`MODEL_TERMS.md`](./MODEL_TERMS.md) を参照してください。

This project's MIT License does not replace the licenses of downloaded third-party components.

## 主なコンポーネント / Principal components

- **ComfyUI** — 固定commit `14b05228cef127ce529bc0c08660770d4af3e9a8`をセットアップ時に取得し、別processとして実行します。License: GNU General Public License v3.0. [Pinned source and license](https://github.com/Comfy-Org/ComfyUI/blob/14b05228cef127ce529bc0c08660770d4af3e9a8/LICENSE)
- **Comfy-Org workflow_templates** — 固定commit `7653f1cdef1d92394b6ef9946018c0a8aa4136b8`をワークフロー設計の参照元としています。H3-STUDIOへvendorせず、セットアップもこのGitリポジトリの独立checkoutを作成しません。License: MIT. [Pinned source and license](https://github.com/Comfy-Org/workflow_templates/blob/7653f1cdef1d92394b6ef9946018c0a8aa4136b8/LICENSE)
- **SageAttention** — upstream License: Apache License 2.0. [Official source and license](https://github.com/thu-ml/SageAttention/blob/main/LICENSE) Windowsではupstream公式binaryではなく、第三者`woct0rdho`による固定wheel `2.2.0+cu130torch2.10.0andhigher.post6`を取得します。[Third-party Windows wheel release](https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post6) セットアップは16,656,067 bytesおよびSHA-256 `1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`を検証します。
- **triton-windows** — version `3.7.1.post27`。License: MIT. [Official project and license](https://github.com/woct0rdho/triton-windows/blob/main/LICENSE)
- **Hugging Face Diffusers** — optional legacy comparison routeのみ。固定commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`。License: Apache License 2.0. [Pinned source and license](https://github.com/huggingface/diffusers/blob/abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc/LICENSE)
- **Qwen3-VL** — MiniMax H3のtext encoderはQwen3-VL-32B系です。MiniMax公式ライセンスはこのencoderをApache License 2.0として明記しています。Qwen側のライセンスとモデルカードも確認してください。[Qwen3-VL license](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE) / [Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- **PyTorch / torchvision / torchaudio** — セットアップ時に公式wheelを取得します。PyTorch本体はBSD-style licenseで、各projectの同梱noticeも適用されます。[PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)

## その他の依存関係 / Other dependencies

[`requirements.webui.txt`](./requirements.webui.txt)、[`requirements.comfy.txt`](./requirements.comfy.txt)、[`requirements.runtime.txt`](./requirements.runtime.txt) に記載された各Python packageは、それぞれ独自のライセンスで配布されています。H3-STUDIOはそれらをGitへvendorせず、隔離virtual environmentへpackage indexからインストールします。再配布や製品組み込みを行う場合は、実際にインストールされたversionのmetadata、license、NOTICEを確認してください。

本書は依存関係の把握を助けるための非網羅的な一覧であり、法的助言ではありません。
