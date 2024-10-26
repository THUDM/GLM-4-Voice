# GLM-4-Voice
GLM-4-Voiceは、Zhipu AIが提供するエンドツーエンドの音声モデルです。GLM-4-Voiceは、中国語と英語の音声を直接理解し生成することができ、リアルタイムの音声対話を行い、ユーザーの指示に基づいて感情、イントネーション、話速、方言などの属性を変更することができます。

## モデルアーキテクチャ

![Model Architecture](./resources/architecture.jpeg)
GLM-4-Voiceの3つのコンポーネントを提供します：
* GLM-4-Voice-Tokenizer: [Whisper](https://github.com/openai/whisper)のエンコーダ部分にベクトル量子化を追加して訓練され、連続音声入力を離散トークンに変換します。1秒の音声は12.5個の離散トークンに変換されます。
* GLM-4-Voice-9B: [GLM-4-9B](https://github.com/THUDM/GLM-4)に基づいて音声モダリティで事前訓練および整合され、離散化された音声を理解し生成することができます。
* GLM-4-Voice-Decoder: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)に基づいて再訓練されたストリーミング推論をサポートする音声デコーダで、離散音声トークンを連続音声出力に変換します。10個の音声トークンで生成を開始でき、対話の遅延を減らします。

詳細な技術レポートは後日公開予定です。

## モデルリスト
|         モデル         | タイプ |      ダウンロード      |
|:---------------------:| :---: |:------------------:|
| GLM-4-Voice-Tokenizer | 音声トークナイザ | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-tokenizer) |
|    GLM-4-Voice-9B     | チャットモデル |  [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-9b)
| GLM-4-Voice-Decoder   | 音声デコーダ |  [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-decoder)

## 使用方法
直接起動できるWebデモを提供しています。ユーザーは音声またはテキストを入力し、モデルは音声とテキストの両方で応答します。

![](resources/web_demo.png)

### 準備
まず、リポジトリをダウンロードします
```shell
git clone --recurse-submodules https://github.com/THUDM/GLM-4-Voice
cd GLM-4-Voice
```
次に、依存関係をインストールします。
```shell
pip install -r requirements.txt
```
Decoderモデルは`transformers`を介して初期化をサポートしていないため、チェックポイントを別途ダウンロードする必要があります。

```shell
# Gitモデルのダウンロード、git-lfsがインストールされていることを確認してください
git clone https://huggingface.co/THUDM/glm-4-voice-decoder
```

### Webデモの起動
まず、モデルサービスを起動します
```shell
python model_server.py --model-path glm-4-voice-9b
```

次に、Webサービスを起動します
```shell
python web_demo.py
```
その後、http://127.0.0.1:8888でWebデモにアクセスできます。

### 既知の問題
* Gradioのストリーミングオーディオ再生は不安定になることがあります。生成が完了した後、対話ボックス内のオーディオをクリックすると音質が向上します。

## 例
GLM-4-Voiceのいくつかの対話例を提供しています。感情制御、話速の変更、方言生成などが含まれます。（例は中国語です。）

* 優しい声でリラックスするようにガイドしてください

https://github.com/user-attachments/assets/4e3d9200-076d-4c28-a641-99df3af38eb0

* 興奮した声でサッカーの試合を解説してください

https://github.com/user-attachments/assets/0163de2d-e876-4999-b1bc-bbfa364b799b

* 哀れな声で幽霊の話をしてください

https://github.com/user-attachments/assets/a75b2087-d7bc-49fa-a0c5-e8c99935b39a

* 東北方言で冬の寒さを紹介してください

https://github.com/user-attachments/assets/91ba54a1-8f5c-4cfe-8e87-16ed1ecf4037

* 重慶方言で「葡萄を食べても皮を吐かない」と言ってください

https://github.com/user-attachments/assets/7eb72461-9e84-4d8e-9c58-1809cf6a8a9b

* 北京アクセントで早口言葉を朗読してください

https://github.com/user-attachments/assets/a9bb223e-9c0a-440d-8537-0a7f16e31651

  * 話速を上げてください

https://github.com/user-attachments/assets/c98a4604-366b-4304-917f-3c850a82fe9f

  * さらに速く

https://github.com/user-attachments/assets/d5ff0815-74f8-4738-b0f1-477cfc8dcc2d

## 謝辞
このプロジェクトの一部のコードは以下から提供されています：
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [GLM-4](https://github.com/THUDM/GLM-4)
