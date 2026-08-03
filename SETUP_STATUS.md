# SETUP STATUS

## Current ComfyUI route

- 状態: ComfyUI native backendへの移行完了。browser UI→private pinned ComfyUI child→H.264/AAC出力・全decodeまで実E2E成功。旧Diffusers結果は比較記録として保持
- 公開確認: 2026-08-03 11:35 JST、公式Hugging Faceで匿名取得可能
- ModelScope公開確認: revision `29139ad62f28479297e305d690ee1521042133d4`、選択60パス存在、READMEを除く59ファイルのサイズ・SHA-256がHugging Face固定revisionと一致
- 固定取得revision: `af0fe5abe6fd50d632b65a82fef321c4c5c1f249`（初回公開commit）
- 最終再確認時の公式head: `9d710aedb174fa4448fcdeeb9a542f11ab52209a`（固定revisionとの差分はREADMEのみ）
- ライセンス: MiniMax H3 Community License Agreement（日本は適用地域、EU／英国／韓国／米国は除外）
- 対象variant: FL2VA＋Ref2VA（Omni）
- 通常生成方式: H3 Studio custom UI＋private ComfyUI native H3 backend。各requestでJobManagerが動的loopback portへfresh childを起動し、`--cache-none`で前jobのmodel stateを持ち越さない。Web UI→worker／worker→ComfyUIの2段をWindows Job Objectへ所属させ、親が先にexitしても`KILL_ON_JOB_CLOSE`で子孫をOS回収。固定8188番では常駐させない
- ComfyUI SHA: `14b05228cef127ce529bc0c08660770d4af3e9a8`
- workflow templates参照元SHA: `7653f1cdef1d92394b6ef9946018c0a8aa4136b8`（設計の出典。通常setupでは未使用のcheckoutを作らない）
- Comfy向けモデル: `Comfy-Org/MiniMax-H3` revision `0543966fbdce5ba05709a8f2031c94bdba629b4a`
- Legacy Diffusers SHA: `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
- GPU: NVIDIA GeForce RTX 5090 32GB
- RAM: 約253GiB
- Cドライブ空き: 約3.48TB（構築開始時）
- Python環境: Web UI `.venv`とComfyUI `.comfy-venv`を分離。Comfy側はPython 3.12.13、torch 2.13.0+cu130／torchvision 0.28.0+cu130／torchaudio 2.11.0+cu130をCUDA 13.0 indexから`--no-deps`で固定し、torchao 0.17.0を使用。CUDAと32kHz audio resampleを確認済み
- ComfyUI上流実装: `14b05228cef127ce529bc0c08660770d4af3e9a8`をdetached checkoutで固定済み
- ComfyUIモデル取得: 5/5、63,440,965,087 bytes（約59.08GiB）
- ComfyUIモデル検証: ローカルSHA-256／byte数が固定HF revisionのLFS SHA-256／sizeと全件一致
- ComfyUI model lock: `comfy_models.lock.json`
- ComfyUI起動前検査: `setup_comfy.ps1 -VerifyOnly -SkipModelHash`成功（固定checkout 2件、CUDA、torchao、32kHz audio resample、SageAttention kernel、5ファイル／63,440,965,087 bytes）。通常起動は同検査を実行し、完全model SHA再検査は明示時のみ
- ComfyUI FL2VA: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`、20,970,379,616 bytes、SHA-256 `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`
- ComfyUI Ref2VA: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`、20,970,379,616 bytes、SHA-256 `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`
- ComfyUI Qwen: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`、15,687,142,551 bytes、SHA-256 `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`
- ComfyUI audio VAE: `minimax_h3_audio_vae_fp32.safetensors`、605,254,808 bytes、SHA-256 `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`
- ComfyUI video VAE: `minimax_h3_video_vae_fp16.safetensors`、5,207,808,496 bytes、SHA-256 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`
- EasyCache: native node。初期値OFF、保守的0.20／高速0.30、sampling 20%～90%、12 steps未満は自動OFF。Sage定常OFF 39.01秒に対し0.20併用39.67秒で、8/20 skip・表示1.67×でもPrompt総時間は同等。近似による映像・音声差もあり、任意選択としてUIへ残す
- SageAttention: `2.2.0+cu130torch2.10.0andhigher.post6`＋`triton-windows 3.7.1.post27`を標準ON。固定Windows wheel 16,656,067 bytes／SHA-256 `1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`をsetupが取得・検証。小型kernel smokeはfinite／repeat equal、SDPA比cosine 0.999344。fallbackは起動前に`$env:H3_ATTENTION_BACKEND='pytorch'`。fallback時はSage import／version／kernel検査を意図的にskipし、PyTorch経路の起動前検査成功を実機確認済み
- Browser E2E: `320×192`／124 frames／Draft 8／EasyCache OFF、初回PyTorchを含むtotal 86.031秒。5.167秒、H.264 124 frames＋AAC 331,776 total samples（165,888／channel）／32kHz stereoを全decode
- Sage標準browser E2E: fresh private childでtotal 44.422秒（Comfy prompt 36.51秒、起動6.813秒、denoise 19秒、Video VAE 6.10秒、全decode検証0.110秒）。`backend=comfy`／`attention_backend=sage`をjobへ保存し、UIにも表示
- 実キャラクター2画像Omni E2E: 各1448×1086の設定画を`<Picture 1>`／`<Picture 2>`、高精度`max`で入力。Job Object／listener ownership安定化後の最終試験は`320×192`／124 frames／Draft 8／EasyCache OFF、total 34.375秒（Comfy prompt 28.047秒、起動6.079秒、検証0.125秒）。5.167秒、H.264 124 frames＋AAC 331,776 total samples（165,888／channel）／32kHz stereoを全decode。2人の主要な外見上の差を保った動きを目視確認。低解像度Draftなので細部同一性の品質保証には使用しない
- PyTorch比較: `640×384`／124 frames／20 stepsの初回Prompt 231.31秒（denoise 36.6秒、Video VAE 180.66秒）。別runはPrompt 117.88秒（EasyCache 7/20 skip・表示1.54×、Video VAE 63.34秒）
- Sage固定server A/B: 初回compile込みPrompt 54.36秒、定常run 39.01秒。Sage＋EasyCache 0.20は39.67秒（denoise 14秒、8/20 skip・表示1.67×）でOFFと総時間同等
- Sage出力検証: 全出力H.264 124 frames＋AACを全decode、黒画面／ノイズなし。同seed通常版と目視ほぼ同等だが、数値的・byte単位の完全一致とは扱わない

## Legacy Diffusers assets and smoke tests

- Legacy上流実装: Diffusers `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` をdetached checkoutで固定済み
- 公式重みダウンロード: 完了（60/60、144,051,067,662 bytes）
- 公式ファイル検証: サイズ・LFS SHA-256全件一致、エラー0
- Diffusers変換: 完了（Transformer 61.73GiB、video VAE 9.70GiB、audio VAE 0.56GiB）
- 変換manifest: 58ファイル、144,051,142,551 logical bytes、SHA-256記録済み
- 共有Qwen／processor／tokenizer: 公式ファイルへのNTFSハードリンクで重複保存を回避
- 公式Ref2VA追加取得: 16ファイル、66,280,525,570 bytes、サイズ・LFS SHA-256全件一致、エラー0
- Ref2VA変換: `transformer_ref` 15ファイル、66,280,569,783 bytes、SHA-256記録済み
- Ref2VA manifest: `artifacts/official_ref2va_manifest.json`／`artifacts/converted_ref2va_manifest.json`
- Ref2VA追加ストレージ: 公式＋変換済みで約123.46GiB
- スモークテスト: 成功（320x192、124 frames、2 sigma points、seed 42）
- スモーク出力: H.264 video 124 frames＋AAC audio 331,776 samples、5.175秒、両ストリーム全decode成功
- スモーク所要時間: 221.62秒（denoise 1 forwardは5.14秒）
- スモーク初回試行: 重みロード前に停止。PR文書の`low_cpu_mem_usage=False`を同SHAの量子化ローダーが拒否する不整合を確認。サポートされる`True`へ修正し、再試行は成功

## H3 Studio

- H3 Studio Web UI: `http://127.0.0.1:7863` localhost限定、Text／Image／Frames／Omni、順番付き参照、進捗リング、キャンセル、履歴、再生、ダウンロード、出力フォルダ表示
- H3 Studio Web境界: numeric loopback以外のHost、cross-site `Sec-Fetch-Site`、専用`X-H3-Studio-Request`ヘッダーのない更新、Hostと一致しないOriginを拒否。CSP／no-store／frame拒否ヘッダーも付与し、DNS rebinding・cross-site GPU job投入を防止
- H3 Studio upload境界: `/api/jobs`の宣言済みaggregate bodyは8GiB、個別素材は2GiB上限。Omniの12件／画像9／動画3／音声3制限と非Omni参照拒否をジョブディレクトリ作成前に検査
- H3 Studio process境界: Windows Job Object `KILL_ON_JOB_CLOSE`を2段で使用。親先行exit後の孤児回収、終了冪等性、ProcessJob作成失敗時のspawn禁止、cancel／runner競合で次engineを誤停止しないことを実機・unit testで確認。private Comfy health後はlistener PIDがspawn tree内であることも検査
- H3 Studio参照UI: 素材タイプ別の公式Picture／Video／Audioタグ自動採番、並べ替え再採番、クリック挿入、音声付き動画のAudio番号説明
- H3 Studio参照精度UI: Omni専用に`高速（match）`／`高精度（max）`を選択。matchは生成canvas相当の総画素へdownscale-only、maxはupscaleせず短辺2048上限。選択値をrequest／jobへ保存し、再利用時に復元
- H3 Studio進捗UI: Qwen解析／参照VAE／レイアウト／denoise／映像復元／音声復元／MP4化を分離。実イベント間だけ次段階未満の上限付き推定を表示
- H3 Studioプロンプト再利用: 成功・失敗・キャンセルを含む永続ジョブ履歴から最大20件の重複なしプロンプトと音響指示を復元
- H3 Studio音響UI: 音の主役、台詞・声質、環境音・効果音、BGM方針を入力し、公式例に沿う`overall_soundscape`／`non_diegetic_music`へ合成。最終出力音量はComfyUI core `AudioAdjustVolume`のraw dBゲインとして適用し、normalization／clipping preventionは行わない。+dBは元peak次第でclipし得る。各設定を永続ジョブ履歴へ保存・復元
- H3 Studio負荷UI: `960×544`・約5秒・Draftを基準に出力側の相対負荷を表示。Omni参照は倍率外の追加負荷として明記し、軽量プレビュー設定ボタンを提供
- Comfy連携unit tests: 52件成功、Windowsでdirectory symlinkを作れない環境向け1件のみskip。ResourceWarningをerror化して成功。client／loopback・upload境界／server request永続化／private child command・listener所有権・Job Object孤児回収・cancel race／EasyCache／Sage-PyTorch fallback／match-max／全4mode workflowを検証
- 音声仕様確認: 公開入力に独立volume／voice strength／audio guidanceはなく、映像と32kHzステレオ音声を共有Transformerで共同生成
- 速度差確認: Diffusers文書は960×544を1344×768より約2.3倍／step高速と記載。公式初回OSSはfull attentionのみ、sparse attentionは今後公開予定。公式SGLangのconsumer最速検証は2×RTX 5090であり、単一5090 CPU offloadとは非同条件

## Legacy Diffusers measurements

- Web UI i2v実生成: 成功、215.86秒、5.175秒H.264＋AAC、124 video frames／331,776 audio samplesを全decode
- Web UI Omni実生成: 成功、234.42秒、5.175秒H.264＋AAC、124 video frames、画素標準偏差57.79、331,776 audio samples、audio RMS 0.00517
- Qwen障害原因: 固定Diffusers SHAがH3に必要な50層目中間出力のために64層すべてを実行し、65 hidden statesを保持。1448×1086画像2枚を短辺2048へupscaleした旧経路は10,880 vision tokens／sequence 11,206で、hidden保持だけでBF16約6.95GiB
- Qwen安定化: 正式マージ済みSGLang／ComfyUI実装に合わせ、checkpointの64層を構築してから削らず最初から50層だけをロード。final norm／LM headなし、`output_hidden_states=False`、SDPA、同期block offloadへ変更
- Ref2VA画像安定化: マージ済みComfyUIの品質優先`max`方式に合わせ、upscaleせず短辺2048を上限に32px整列。実画像2枚は各1448×1086→1440×1088、合計3,060 vision tokens／sequence 3,386
- Qwen構築ベンチ: checkpointの1,058 weight群中904群をロード、25.143秒（旧64層構築33.251秒）。後半14層の`UNEXPECTED`表示は意図した未使用weightの読み飛ばし
- Qwen実画像ベンチ: 同一workerで25.776秒／15.509秒、CUDA peak allocated 7.524GiB／7.522GiB、NaN／Infなし、連続runのembedding／token tagはbyte単位で完全一致
- Ref2VA安定化E2E: 同じ2画像・同じprompt、320×192、124 frames、2 grid pointsを同じpipelineで2回連続成功。各5.175秒H.264 124 frames＋AAC 331,776 samplesを全decode、映像標準偏差62.6301、audio RMS 0.0218702／peak 0.1550
- Ref2VA再現性: 2本とも208,812 bytesで、MP4／previewともbyte単位で完全一致
- 冷間ロード実測: process peak RSS 217.28GiB／peak paged memory 239.69GiB（旧64層構築peak RSS 226.13GiB）。安定化ゲートは空き物理RAM 225GiB、commit余力300GiB、通常空きVRAM 24GiB
- 高負荷ゲート: 画素数×framesが2.5億以上では空きVRAM 29GiBを要求し、pipeline再利用時もdevice全体の空きを再確認
- 最大設定実測: 1344×768・345 frames・2 grid pointsはQwen修正後もTransformer denoise 1 forwardが12分48秒超、VRAM最大約31.7GiBで手動停止。full attention本体の負荷であり、20 grid pointsを単一5090の20分経路とは扱わない
- VAE配置: video VAEはleaf-level CPU offload、audio VAEはleaf offloadでdecoder device mismatchを起こすため検証済みのGPU常駐（約0.56GiB）
- モデル切替: FL2VA↔Ref2VA時に生成ワーカーPIDが変わることを検証済み。同variant連続生成はロード済みモデルを再利用
- 実測メモリ: 不完全な同一プロセス内切替ではページファイル約246GiBまで増加したため不採用。ワーカー完全再起動後は空きRAM約230GiBまで回復

## Repository

- 公開対象: 独自コード、設定、lock、ドキュメントのみ。モデル重み、参照素材、生成物、仮想環境、上流checkout、cacheはGit対象外
- 公開監査: 配布対象48ファイル約430KiB、10MiB超ファイルおよびモデル／動画／画像／音声binaryなし。CPU-only配布検査6件成功
- 最新unit tests: 58件成功、Windowsでdirectory symlinkを作れない環境向け1件のみskip
