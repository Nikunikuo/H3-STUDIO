# H3 Studio

MiniMax H3をWindowsのローカルWeb UIから使う、動画＋同期音声生成スタジオです。Text、Image、開始／終了Frames、複数参照を扱うOmniに対応し、固定revisionのComfyUIバックエンドをH3 Studio自身が起動・停止します。普段の生成でPowerShellコマンドやComfyUIノードを操作する必要はありません。

> [!IMPORTANT]
> モデル重みはこのGitリポジトリに含まれません。初回セットアップが`Comfy-Org/MiniMax-H3`の固定revisionから必要な5ファイルだけ（約59.08GiB）を取得し、サイズとSHA-256を検証します。MiniMax H3 Community Licenseには適用地域、用途、出力、再配布、商用利用に関する制限があります。セットアップ前に[モデル利用条件](./MODEL_TERMS.md)と[公式ライセンス原文](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/af0fe5abe6fd50d632b65a82fef321c4c5c1f249/LICENSE)を必ず確認してください。

## Quick Start

### 対応環境

- Windows 11 64-bit
- 検証済みGPU: desktop NVIDIA GeForce RTX 5090 32GB。セットアップはCUDA GPUのVRAM 30GiB以上を要求しますが、5090以外は未検証です
- 検証済みRAM: 約253GiB。量子化済みComfyUI経路の最低RAMは未確立なので、64／128GiBでの動作は保証していません
- [64-bit Python 3.12](https://www.python.org/downloads/release/python-31210/)（Python 3.13は非対応。`winget install --exact --id Python.Python.3.12`でも導入可能）
- Git for Windows、対応NVIDIA driver、インターネット接続
- セットアップ開始時に約100GiBの空き容量を推奨。モデル約59.08GiBに加え、ComfyUI環境、取得cache、生成物、参照素材を保存します

PythonのCUDA wheelにruntimeが含まれるため、CUDA Toolkitを別途インストールする必要はありません。NVIDIA driverはCUDA 13.0 runtimeを扱える必要があり、セットアップ中に実CUDA処理まで検査します。管理者権限やシステムPythonの変更は行いません。Windowsの長いpath問題を避けるため、空きのあるドライブの`C:\AI\H3-STUDIO`や`D:\AI\H3-STUDIO`のような短い場所へのcloneを推奨します。

### 初回だけ

PowerShellで空きのある保存先へ移動し、cloneします。

```powershell
Set-Location C:\AI
git clone https://github.com/Nikunikuo/H3-STUDIO.git
Set-Location .\H3-STUDIO
.\Setup-H3-Studio.cmd
```

ライセンス原文を確認し、利用資格と同意を確認できる場合だけ`ACCEPT`と入力してください。セットアップはPython 3.12を自動検出し、Web UI、固定ComfyUI、量子化済みモデル、SageAttentionを隔離環境へ構築します。途中で通信が切れても同じファイルを再実行すれば取得を再開できます。

### 2回目以降

エクスプローラーで`Start-H3-WebUI.cmd`をダブルクリックするだけです。ComfyUIを別途起動する必要はありません。H3 Studioが必要なprivate ComfyUI childを自動管理し、ブラウザで<http://127.0.0.1:7863>を開きます。

## H3 Studio Web UI（推奨）

初回セットアップ後は、エクスプローラーで`Start-H3-WebUI.cmd`をダブルクリックすると、ローカル専用Web UIがブラウザで開きます。

- URL: <http://127.0.0.1:7863>
- 接続範囲: このPCのlocalhostのみ。外部公開・外部アップロードなし
- Text: 文章から動画＋同期音声
- Image: 開始画像からi2v＋同期音声
- Frames: 開始画像＋終了画像から動画＋同期音声
- Omni: 順番付きの画像／動画／音声参照から動画＋同期音声（公式Ref2VA）
- Omni素材へ公式`<Picture 1>`／`<Video 1>`／`<Audio 1>`タグを自動採番。タグクリックでプロンプトへ挿入
- Omni参照画像の解析精度を`高速（match）`／`高精度（max）`から選択
- 生成スタイル、音声・音響、解像度、長さ、品質、seedを簡単設定
- 音の主役、台詞・声質、環境音・効果音、BGM方針を公式推奨の音響プロンプト構造へ自動変換
- MP4書き出し直前の最終出力音量を−12～+6dBから選択（実波形へraw dBゲインを適用）
- 出力サイズ・長さ・denoise回数から相対負荷を表示し、ワンクリックで軽量プレビュー設定へ変更
- モデル準備、denoise、デコード、MP4書き出しを進捗リングで表示
- 生成履歴、過去プロンプト＋音響指示の再利用、動画再生、ダウンロード、出力フォルダ表示

黒いPowerShell画面はローカル生成サーバー本体なので、利用中は閉じないでください。ブラウザのタブを閉じてもサーバーは動き続けます。終了するときはPowerShellで`Ctrl+C`を押します。

ブラウザが接続するのはFastAPI製のH3 Studio（`127.0.0.1:7863`）だけです。生成ジョブが始まると、H3 Studioが固定SHAのComfyUIをprivate childとして動的なloopback portへ起動し、終了・キャンセル時には同じ管理単位で回収します。ComfyUIを固定`8188`番で常駐させたり、ComfyUI標準画面を外部公開したりはしません。

Web境界でもnumeric loopback以外の`Host`、cross-site browser request、専用ローカルヘッダーのない更新操作を拒否します。これはDNS rebindingや、別サイトから勝手にGPUジョブを投入されることを防ぐためです。ジョブ本文は宣言サイズ8GiB、個別ファイル2GiBを上限とし、Omniの件数・種別制限はジョブ用ディレクトリへコピーする前に検査します。画面からの通常操作には追加設定は不要です。

参照素材とジョブ情報は`webui_data/`、完成動画は`outputs/webui/`へ保存され、どちらもGit対象外です。これらは自動削除されないため、容量を空ける場合はH3 Studioを終了してから不要な内容を削除してください。`models/`は再取得に約59.08GiB必要なので、履歴整理の対象と取り違えないでください。GPUジョブは安全のため1本ずつ実行されます。モデルが返す実イベントを進捗の基準にし、イベント間だけ次段階の上限を超えない推定値を小数表示します。denoiseは実際の完了ステップ数を基準に、長い1 stepの途中だけ推定表示します。円内が`ESTIMATE`の間は推定、`PROGRESS`はモデルから届いた実値です。

各生成はfreshなprivate ComfyUI childを使い、完了・失敗・キャンセル後に子孫processごと回収します。WindowsではWeb UI→worker、worker→ComfyUIの2段を`KILL_ON_JOB_CLOSE`付きJob Objectへ入れるため、親が先に異常終了した場合も残った子孫をOS側で回収します。キャンセル時は対象engineの所有権をlock内で確定してから停止するため、直後に始まった次ジョブを誤停止しません。通常workerは安定性と後始末を優先して`--cache-none`で起動するため、別ジョブへロード済みモデルを持ち越しません。約59.08GiBの量子化済みモデル群を毎回準備する時間はかかりますが、前ジョブのGPU／RAM状態や固定portを引きずらない構成です。

### 現在のComfyUIバックエンド

- ComfyUI: `14b05228cef127ce529bc0c08660770d4af3e9a8`
- 公式workflow templates: `7653f1cdef1d92394b6ef9946018c0a8aa4136b8`
- Comfy-Orgモデル: `Comfy-Org/MiniMax-H3` revision `0543966fbdce5ba05709a8f2031c94bdba629b4a`
- Transformer: FL2VA／Ref2VAのpruned int8 convrotを用途別に使用
- text encoder: MiniMax H3用Qwen3-VL-32B NVFP4/AWQ
- VAE: video FP16＋audio FP32
- モデル5ファイル: 合計63,440,965,087 bytes（約59.08GiB）。個別の期待byte数とSHA-256は`comfy_models.lock.json`へ固定
- Python: `.comfy-venv`。Web UI本体の`.venv`やlegacy Diffusersとは分離。torch 2.13.0+cu130、torchvision 0.28.0+cu130、torchaudio 2.11.0+cu130、torchao 0.17.0を固定
- attention: SageAttention `2.2.0+cu130torch2.10.0andhigher.post6`＋triton-windows `3.7.1.post27`を標準ON

Omni画像はUIの「参照画像の解析精度」で選べます。初期値の`高速（match）`は元画像を拡大せず、縦横比を維持したまま生成canvasと同程度の総画素数まで必要な場合だけ縮小し、速度比較と安定性を優先します。`高精度（max）`もupscaleは行わず、短辺2048pxを上限に原寸付近のディテールを使うため、人物や製品の同一性を確認するときに向きます。ただし参照tokenが全sampling stepへ入るため、matchより数倍遅くなり得ます。選択値はジョブ履歴へ保存され、プロンプト再利用時にも復元されます。

EasyCacheはComfyUI native nodeによる近似で、初期値は`OFF`です。`保守的`はreuse threshold `0.20`、`高速`は`0.30`で、どちらもsampling区間20%～90%だけを対象にします。12 steps未満のDraftでは自動OFFです。SageAttention定常runでは、EasyCache OFFのPrompt実行39.01秒に対して0.20併用は39.67秒で、8/20 stepsをskipしても総時間は同等でした。映像・音声がわずかに変化し得る近似でもあるため、通常はOFFのまま使い、個別素材で効果を確認したい場合だけ有効にします。

SageAttentionは実動画A/Bを通過したため標準ONです。setupはWindows wheel（16,656,067 bytes、SHA-256 `1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`）を固定URLから取得・検証し、ComfyUIへ`--use-sage-attention`を渡します。全Sage出力は124 framesのH.264＋AACを全decodeでき、黒画面やノイズはなく、同seedのPyTorch版と目視でほぼ同等でした。ただし数値的・byte単位で完全同一とは扱いません。互換性問題を切り分ける場合は、起動前に`H3_ATTENTION_BACKEND=pytorch`を指定して公式PyTorch attentionへ戻せます。

### 実機E2Eと速度確認

ブラウザのH3 Studioからprivate pinned ComfyUI childを起動する通常経路で、`320×192`／124 frames／Draft 8／EasyCache OFFを生成し、初回PyTorch実行を含む総時間86.031秒で完了しました。出力は5.167秒、H.264 124 frames＋AAC 331,776 total samples（165,888／channel）／32kHz stereoで、両streamを全decode済みです。

Sage標準化後も同じ通常ブラウザ経路を再実行し、fresh private childの起動・固定資産検証を含む総時間44.422秒（Comfy prompt 36.51秒）で完了しました。内訳はComfy起動6.813秒、denoise 19秒、Video VAE 6.10秒、出力全decode検証0.110秒です。ジョブには`backend=comfy`と`attention_backend=sage`が保存され、画面にも`Backend comfy · SageAttention`と表示されます。

実運用相当のキャラクター設定画2枚でもOmni経路を確認しました。各1448×1086の画像を`<Picture 1>`／`<Picture 2>`として高精度`max`で同時に入力し、`320×192`／124 frames／Draft 8／EasyCache OFFを、最終安定化後の再試験では総時間34.375秒（Comfy prompt 28.047秒、起動6.079秒、全decode検証0.125秒）で完了しました。出力は5.167秒、H.264 124 frames＋AAC 331,776 total samples（165,888／channel）／32kHz stereoを全decode済みです。2人の主要な外見上の差を保った動きが成立しました。これは低解像度Draftの動作確認であり、細部の同一性を保証する品質評価とは扱いません。

`640×384`／124 frames／20 stepsの比較値は次のとおりです。Sageの3本は固定server上でmodel reloadを除いてattentionとcacheの差を測った値なので、通常UIの総所要時間ではありません。

| attention／cache | Comfy Prompt実行 | denoise | Video VAE | 備考 |
|---|---:|---:|---:|---|
| PyTorch 初回 | 231.31秒 | 36.6秒 | 180.66秒 | 初回処理を含む |
| PyTorch 別run＋EasyCache | 117.88秒 | — | 63.34秒 | 7/20 skip、表示上1.54× |
| Sage 初回 | 54.36秒 | — | — | 初回compileを含む |
| Sage 定常run／OFF | 39.01秒 | — | — | 標準構成 |
| Sage＋EasyCache 0.20 | 39.67秒 | 14秒 | — | 8/20 skip、表示上1.67×。総時間はOFFと同等 |

旧PyTorch経路でVideo VAEに180.66秒かかった値は正常な基準ではなく、VAE内attentionがこのGPUで遅かった回帰値です。同じ`640×384`／124 framesをSageAttentionへ切り替えた通常ブラウザ試験ではVideo VAEが6.10秒まで短縮しました。またOmniログの最初の`VideoVAE request`は参照画像のencodeであり、次の`AudioVAE request`までの区間にはQwen解析、Transformer準備、denoise、出力video decodeがまとめて含まれます。その区間全体を「VAEデコード時間」とは扱いません。

### 旧Diffusers経路の調査記録（比較用）

以下は固定Diffusers PRを使って原因を切り分けた時点の記録です。通常のH3 Studio生成経路ではありません。

旧Diffusers Ref2VA workerでは、マージ済みComfyUI H3実装の品質優先`max`方式を参考に、画像を無意味に拡大せず、短辺2048pxを上限として32px単位へ整列しました。1448×1086の一般的な画像なら実質的にネイティブの1440×1088となり、補間だけで増えるvision tokenを避けます。旧経路の初回モデル切替はRAM使用量が大きいため、ほかの大規模AI処理との同時実行を避けました。

固定Diffusers SHAの素のRef2VA encoderは、H3が使う50層目の中間出力を得るだけなのにQwenの64層すべてを実行し、65個のhidden stateを保持していました。旧workerは正式マージ済みSGLang／ComfyUI実装と同じく最初の50層だけを実行し、final normとLM headを使わず、未正規化の最終出力1個だけを保持するよう修正しました。QwenはSDPAを明示し、leaf単位ではなく同期decoder block単位でCPU offloadします。これは近似や画質設定ではなく、旧`hidden_states[50]`と同じテンソルを余分な後半14層なしで直接得る安定化です。

旧workerのQwenは64層を一度構築してから削るのではなく、checkpointから必要な50層だけを最初から構築します。そのためロード時に`layers.{50...63}`が`UNEXPECTED`と表示されますが、これは意図的に読み飛ばした未使用14層であり、欠損ではありません。実画像2枚のembeddingが旧`hidden_states[50]`と完全一致し、同一seedの連続生成MP4もSHA-256単位で一致することを確認しています。

Omniの番号は素材タイプごとです。たとえば画像→動画→画像→音声のUI順なら、`<Picture 1>`→`<Video 1>`→`<Picture 2>`→`<Audio 1>`になります。並べ替えるとタグも自動再採番されます。動画に音声トラックがある場合、その音声には内部で別途`<Audio n>`が付きます。

### 音声・音響の指定

H3は映像と音声を別々に後処理するモデルではありません。1つのpacked sequenceを共有33B Transformerでdenoiseし、映像と32kHzステレオ音声を共同生成します。公開モデル入力とComfyUI native H3 nodeには、独立した`audio_prompt`、voice strength、audio guidance scaleはありません。UIの音響欄は、存在しない数値パラメータを装うのではなく、次の内容を元プロンプトとは別の構造へまとめます。

- 音の主役: 会話、環境音、効果音、音楽、静かな音場
- 台詞・声質: 誰が、どんな声で、何を話すか、口の動きとの同期
- 環境音・効果音: room tone、天候、フォーリー、動作に同期する効果音
- BGM: 自動、なし、控えめ、はっきり
- 最終出力音量: −12dB、−6dB、0dB、+3dB、+6dB

内部では公式H3-Context-IRの出力例に合わせて`overall_soundscape:`と`non_diegetic_music:`を組み立てます。音の内容に関する項目は自然言語による生成誘導であり、ミキサーのdB値のような決定的制御ではありません。一方、最終出力音量はComfyUI coreの`AudioAdjustVolume`で、生成後の実波形へ`10^(dB/20)`をそのまま乗算します。normalizationやclipping preventionは行わないため、+3dB／+6dBは元波形のpeak次第でclipし得ます。日本語の会話は公式に安定対応する11言語へ含まれます。Omniの音声ファイルは声質、話速、タイミング、既存BGM等の参照にも使えますが、音声単独では指定できず画像または動画との組み合わせが必要です。

### 速度を比較するときの注意

「15秒」という長さだけでは速度を比較できません。計算量には少なくとも解像度、denoise回数、T2V／FL2VA／Ref2VA、参照素材の数と長さ、量子化、offload、GPU枚数、attention backendが影響します。

- Diffusers PR文書では、`960×544`は学習canvasの`1344×768`より1 stepあたり約2.3倍高速とされています。
- 旧Diffusersの`num_inference_steps=20`は終端0を含むため、実際のTransformer評価は19回です。
- 旧Diffusersの`1344×768`・345 frames・20 grid pointsは、`960×544`・124 frames・Draft 8 grid pointsを基準に、画素数×frame数×評価回数だけでも約14.9倍です。
- 安定化後の旧Diffusers Ref2VA画像は拡大しません。1448×1086は1440×1088となり1枚1,530 vision tokensです。修正前の2720×2048／1枚5,440 tokensと比べ、2枚時は10,880から3,060 tokensへ減ります。
- 初回オープンソース版はfull attentionのみです。MiniMaxの学習・推論で使うnative sparse attentionは今後公開予定と公式モデルカードに記載されています。
- 公式SGLangの最速consumer検証は`2×RTX 5090`＋layerwise offloadです。このPCの`1×RTX 5090`＋CPU group offloadとはハードウェアも実装経路も異なります。
- 現在の通常経路は公式ComfyUIテンプレートと同系統のpruned int8 Transformer／NVFP4-AWQ Qwen encoderです。ここに残す旧Diffusers、SGLang、クラウド／複数GPUの報告時間は実装条件が異なるため相互にそのまま当てはめられません。
- このPCで`1344×768`・345 frames・2 grid points（Transformer評価1回）の同一2画像テストは、Qwen修正後もdenoise 1回だけで12分48秒を超えたため手動停止しました。VRAM使用量は最大約31.7GiBで、Qwen障害とは別にfull-attention Transformer本体の負荷が支配的です。20 grid pointsなら同条件の評価が19回になるため、単一5090の実用的な20分経路ではありません。

当時のWeb UI負荷倍率は`960×544`・約5秒・Draftを1とした、画素数×frame数×denoise回数の相対目安でした。Omni参照解析とattention増加は倍率へ含めず、別の追加負荷として表示しました。

旧Diffusers backendは冷間ロード前に物理RAM 225GiB、Windows commit余力300GiB、通常設定で空きVRAM 24GiBを要求しました。画素数×framesが2.5億以上では空きVRAM 29GiBを要求し、Windows共有GPUメモリへ落ちる前に生成を停止する保護を入れていました。これは現在の量子化済みComfyUI経路の要件ではありません。

## 固定した一次資料

- 公式ModelScope: <https://modelscope.cn/models/MiniMax/MiniMax-H3>
- ModelScope確認revision: `29139ad62f28479297e305d690ee1521042133d4`
- 公式モデル: <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- 固定取得revision: `af0fe5abe6fd50d632b65a82fef321c4c5c1f249`（初回公開commit）
- 最終再確認時の公式head: `9d710aedb174fa4448fcdeeb9a542f11ab52209a`（固定revisionとの差分はREADME 1ファイルのみ。モデル・設定ファイルは同一）
- Diffusers統合PR: <https://github.com/huggingface/diffusers/pull/14355>
- Diffusers revision: `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
- ComfyUI正式対応PR: <https://github.com/Comfy-Org/ComfyUI/pull/15224>
- ComfyUI H3 merge commit: `57500fc5bc92566a63f2046824f522cd55c335ca`
- 通常生成に固定したComfyUI: <https://github.com/Comfy-Org/ComfyUI/tree/14b05228cef127ce529bc0c08660770d4af3e9a8>
- 固定workflow templates: <https://github.com/Comfy-Org/workflow_templates/tree/7653f1cdef1d92394b6ef9946018c0a8aa4136b8>
- Comfy-Orgモデル配布: <https://huggingface.co/Comfy-Org/MiniMax-H3/tree/0543966fbdce5ba05709a8f2031c94bdba629b4a>
- ComfyUI公式H3チュートリアル: <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- SageAttention公式: <https://github.com/thu-ml/SageAttention>
- 固定Windows wheel release: <https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post6>
- SGLang H3対応PR: <https://github.com/sgl-project/sglang/pull/33275>
- SGLang Qwen encoder確認commit: `70fe2e0dd5e69a062fb146d0db35c7ac939f111f`
- SGLang公式H3 cookbook／benchmark: <https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3>

## ライセンス上の注意

H3 Studio独自コードはルートの[MIT License](./LICENSE)で公開します。このMIT Licenseは、MiniMax H3／Comfy-Orgのモデル重み、ComfyUI、SageAttention、Qwen、Diffusers、PyTorch、その他の依存物を再ライセンスしません。

MiniMax H3 Community License Agreementは、EU、英国、韓国、米国を適用地域から除外しています。本セットアップは日本国内のローカル環境で検証しています。利用、複製、実行等を行うことでライセンスへ同意した扱いになり得ます。用途、出力、再配布、第三者提供、商用条件を含む重要事項は[MODEL_TERMS.md](./MODEL_TERMS.md)から公式原文を確認してください。依存物のライセンスは[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)に整理しています。

## 構成

- H3-Base-FL2VA: テキスト→動画＋音声、先頭／末尾フレーム→動画＋音声
- H3-Base-Ref2VA: 参照画像／動画／音声→動画＋音声
- 33B dense single-stream Transformer。公開設定はhidden 5376、50層、56 heads × 128、FFN 14336。
- text encoderはQwen3-VL-32B。H3は50層目直後の未正規化hidden stateを使用します。
- guidanceは重みに蒸留済みで、negative prompt／CFGは使いません。

通常生成に必要なComfy-Org版5ファイルは、FL2VA／Ref2VA／共有Qwen／video VAE／audio VAEを合わせて63,440,965,087 bytes（約59.08GiB）です。ローカル実測SHA-256は固定revisionのLFS SHA-256と全件一致しています。元のMiniMaxAI公式リポジトリ全体は、共有ファイルの重複をLFS SHA-256単位で除いても210,501,560,795 bytes（約196.04GiB）で、こちらはlegacy Diffusers比較を再現する場合だけ必要です。

## セットアップ（通常のComfyUI経路）

通常はQuick Startの`Setup-H3-Studio.cmd`を使ってください。内部では、`setup.ps1`が最小構成のWeb UI用`.venv`を作り、`setup_comfy.ps1`がPython 3.12の`.comfy-venv`、固定ComfyUI checkout、固定revisionの5モデル、固定SageAttention wheelを用意します。不要なworkflow templatesリポジトリはcloneせず、旧Diffusers比較環境も通常セットアップへ混ぜません。モデルのbyte数とSHA-256は`comfy_models.lock.json`へ照らして検証します。巨大重み、wheel cache、仮想環境、上流checkoutはGitへ入りません。

自動化などでモデル取得を伴うセットアップを直接実行する場合は、公式ライセンスを確認したうえで明示スイッチを付けます。全5モデルが既に正しいサイズで存在し、再ダウンロードしない検査・修復ではスイッチを要求しません。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_comfy.ps1 -AcceptMiniMaxH3License
```

起動前と同じ高速検査を単独で行う場合:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_comfy.ps1 -VerifyOnly -SkipModelHash
```

5ファイル約59.08GiBを読み直して完全SHA-256検査する場合は`-SkipModelHash`を外します。通常起動は固定source SHA、隔離runtime、CUDA、5ファイルの厳密なbyte数だけを高速検査します。

SageAttentionを使わずPyTorch attentionで再現・切り分ける場合は、先に稼働中のH3 StudioをそのPowerShellで`Ctrl+C`して終了し、新しいPowerShellで次のように起動します。異なるattention backendの既存serverが動いている場合、起動スクリプトは黙って再利用せず明示的に停止を求めます。

```powershell
$env:H3_ATTENTION_BACKEND = 'pytorch'
.\Start-H3-WebUI.cmd
```

環境変数を指定しない通常起動はSageAttentionです。SageAttention wheelの導入自体を避けて初回構築する場合は、同じ環境変数を設定してから`Setup-H3-Studio.cmd`を実行してください。PyTorch構成では第三者SageAttention wheelとtriton-windowsを取得しません。ブラウザE2Eを再現する最小設定は、Text、`320×192`、約5秒（124 frames）、Draft 8、EasyCache OFFです。

### Legacy Diffusers比較の再構築

原因調査時のraw MiniMaxAI重みと固定Diffusers workerを再現する場合だけ、以下を追加実行します。通常生成には不要です。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -WithLegacyDiffusers
.\.venv\Scripts\python.exe .\scripts\download_official.py
.\.venv\Scripts\python.exe .\scripts\prepare_diffusers.py
.\.venv\Scripts\python.exe .\scripts\download_ref2va.py
.\.venv\Scripts\python.exe .\scripts\prepare_ref2va.py
.\.venv\Scripts\python.exe .\scripts\inference_smoke.py
```

このlegacyダウンロードも固定revisionかつ再開可能で、公式モデルリポジトリ内のPythonコードは取得・実行しません。ModelScopeの選択対象60パスについても、更新READMEを除く59ファイルがHugging Face固定revisionとサイズ・SHA-256一致済みです。

## 5090向け方式

現在の経路は、Comfy-Orgが用意したpruned int8 Transformer（用途別に各約19.53GiB）とNVFP4/AWQ Qwen（約14.61GiB）をComfyUI native H3 nodesで読みます。起動時にBF16の33B TransformerとQwenを丸ごと構築してから量子化する旧経路ではありません。VRAM/RAM配置は固定ComfyUIのmodel managementへ任せ、実動画で検証したSageAttentionを標準使用します。EasyCacheは実測上の総時間短縮がなかったため初期値OFFです。

### Legacy Diffusers検証結果

旧経路はTransformer（BF16約61.7GiB）とQwen3-VL（約62.1GiB）をロード時にint8 weight-only量子化し、CPUからblock単位でstreamしました。video VAEをCPU offload、audio VAEをGPU常駐としたWindows量子化ローダーの冷間構築では、一時的に約217GiB RAMを実測しています。

Diffusers PR #14355のconsumer GPUサンプルは`low_cpu_mem_usage=False`を指定していますが、同じ固定SHAの量子化ローダーはFalseを明示的に拒否します。本リポジトリはローダーがサポートする`True`を使用します。この差異はスモークテストで実動作を確認します。

スモークテストは最小負荷を優先し、320×192、124 frames（約5秒）、2 sigma grid points（1回のモデル評価）、seed 42で動画とステレオ音声を共同生成します。品質評価ではなく、ロード、denoise、video/audio VAE、MP4 mux、音声トラック検証が目的です。

検証済み結果（2026-08-03）:

- 総実行時間: 221.62秒
- denoise: 1 forward、5.14秒
- MP4: 197,930 bytes、5.175秒
- video: H.264、124 framesを全decode、画素標準偏差48.90
- audio: AAC、162 audio frames／331,776 samplesを全decode、RMS非ゼロ
- ロード時RAM一時ピーク: 空き約2.5GiBまで低下後、量子化前テンソル解放で約68GiB以上へ回復
- 実行時GPU: 他アプリ使用込みで空き約22GiBから開始、video VAEもCPU offload

Ref2VA安定化の実画像検証（2026-08-03）:

- 過去に18分57秒～2時間51分41秒Qwen区間を抜けられなかった同一2画像を使用
- 入力2枚: 各1448×1086、前処理後は各1440×1088、合計3,060 vision tokens
- Qwen sequence: 3,386、embedding shape: `1×3386×5120` BF16、NaN／Infなし
- Qwenはcheckpointの1,058 weight群から必要な904群だけを構築。ロード25.143秒（旧64層構築33.251秒）
- Qwen単体を同一workerで2回連続実行: 25.776秒／15.509秒、CUDA peak allocated 7.524GiB／7.522GiB、両runのembedding／token tagが完全一致
- 連続runのembedding／token tagはbyte単位で完全一致
- 64層から旧`hidden_states[50]`を得る方式と、50層で停止する方式の完全一致を単体テストで検証
- 同じ2画像・同じプロンプトのRef2VA全工程を、同じロード済みpipelineで2回連続成功: 320×192、124 frames、2 grid points
- 両MP4は各208,812 bytesで、byte単位で完全一致
- 各MP4は5.175秒、H.264 124 frames＋AAC 331,776 samplesを全decode。画素標準偏差62.6301、audio RMS 0.0218702／peak 0.1550
- 冷間ロード時process peak RSS 217.28GiB（旧64層構築経路226.13GiB）、peak paged memory 239.69GiB

Qwenだけを再検証する場合:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_qwen_ref2va.py --request .\webui_data\jobs\<job-id>\request.json --repeat 2
```

生成品質を確認する場合は`num_inference_steps`を増やし、学習canvasに近い解像度を使用してください。スモーク値の2 grid pointsは実装経路の通過確認専用です。

## Git除外

公式重み、変換済み重み、仮想環境、上流checkout、キャッシュ、生成物、検証ログ、秘密情報はすべてGit対象外です。
