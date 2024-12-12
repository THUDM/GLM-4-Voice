import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
from argparse import ArgumentParser

import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from audio_process import AudioStreamProcessor
from flow_inference import AudioDecoder
import torch
import gradio as gr
from speech_tokenizer.utils import extract_speech_token

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")


def ngrok_proxy(port):
    """
    run `ngrok config add-authtoken $NGROK_TOKEN`
    """
    from pyngrok import ngrok
    import nest_asyncio

    ngrok_tunnel = ngrok.connect(port)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--server-addr", type=str,
                        default="http://localhost:10000", help="server address")
    parser.add_argument("--ngrok", action='store_true', help="use ngrok proxy")
    args = parser.parse_args()

    if args.ngrok:
        ngrok_proxy(args.port)

    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    glm_tokenizer = None
    device = "cuda"
    audio_decoder: AudioDecoder = None
    whisper_model, feature_extractor = None, None

    def initialize_fn():
        global audio_decoder, feature_extractor, whisper_model, glm_model, glm_tokenizer
        if audio_decoder is not None:
            return

        # GLM
        glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        # Flow & Hift
        audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                     hift_ckpt_path=hift_checkpoint,
                                     device=device)

        # Speech tokenizer
        whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

    def clear_fn():
        return [], [], '', '', '', None, None

    def inference_fn(
            temperature: float,
            top_p: float,
            max_new_token: int,
            input_mode,
            audio_path: str | None,
            input_text: str | None,
            history: list[dict],
            previous_input_tokens: str,
            previous_completion_tokens: str,
    ):

        if input_mode == "audio":
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise gr.Error("No audio tokens extracted")
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        else:
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            user_input = input_text
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        with torch.no_grad():
            response = requests.post(
                f"{args.server_addr}/generate_stream",
                data=json.dumps({
                    "prompt": inputs,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_token,
                }),
                stream=True
            )
            text_tokens, audio_tokens = [], []
            audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            end_token_id = glm_tokenizer.convert_tokens_to_ids('<|user|>')
            complete_tokens = []
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
            this_uuid = str(uuid.uuid4())
            tts_speechs = []
            tts_mels = []
            prev_mel = None
            is_finalize = False
            block_size_list = [25, 50, 100, 150, 200]
            block_size_idx = 0
            block_size = block_size_list[block_size_idx]
            audio_processor = AudioStreamProcessor()
            for chunk in response.iter_lines():
                token_id = json.loads(chunk)["token_id"]
                if token_id == end_token_id:
                    is_finalize = True
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                    if block_size_idx < len(block_size_list) - 1:
                        block_size_idx += 1
                        block_size = block_size_list[block_size_idx]
                    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)

                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(device),
                                                                  prompt_feat=prompt_speech_feat.to(device),
                                                                  finalize=is_finalize)
                    prev_mel = tts_mel

                    audio_bytes = audio_processor.process(
                        tts_speech.clone().cpu().numpy()[0], last=is_finalize)

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    if audio_bytes:
                        yield history, inputs, '', '', audio_bytes, None
                    flow_prompt_speech_token = torch.cat(
                        (flow_prompt_speech_token, tts_token), dim=-1)
                    audio_tokens = []
                if not is_finalize:
                    complete_tokens.append(token_id)
                    if token_id >= audio_offset:
                        audio_tokens.append(token_id - audio_offset)
                    else:
                        text_tokens.append(token_id)
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        history.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
        history.append({"role": "assistant", "content": glm_tokenizer.decode(
            text_tokens, ignore_special_tokens=False)})
        yield history, inputs, complete_text, '', None, (22050, tts_speech.numpy())

    def update_input_interface(input_mode):
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]

    # Create the Gradio interface
    with gr.Blocks(title="GLM-4-Voice Demo", fill_height=True) as demo:
        with gr.Row():
            temperature = gr.Number(
                label="Temperature",
                value=0.2
            )

            top_p = gr.Number(
                label="Top p",
                value=0.8
            )

            max_new_token = gr.Number(
                label="Max new tokens",
                value=2000,
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(["audio", "text"], label="Input Mode", value="audio")
                audio = gr.Audio(
                    label="Input audio",
                    type='filepath',
                    show_download_button=True,
                    visible=True)
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="Enter your text here...",
                    lines=2,
                    visible=False)

            with gr.Column():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                output_audio = gr.Audio(label="Play", streaming=True,
                                        autoplay=True, show_download_button=False)
                complete_audio = gr.Audio(
                    label="Last Output Audio (If Any)",
                    show_download_button=True)

        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(
                label=f"Input Tokens",
                interactive=False,
            )

            completion_tokens = gr.Textbox(
                label=f"Completion Tokens",
                interactive=False,
            )

        detailed_error = gr.Textbox(
            label=f"Detailed Error",
            interactive=False,
        )

        history_state = gr.State([])

        respond = submit_btn.click(
            inference_fn,
            inputs=[
                temperature,
                top_p,
                max_new_token,
                input_mode,
                audio,
                text_input,
                history_state,
                input_tokens,
                completion_tokens,
            ],
            outputs=[
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio
            ]
        )

        respond.then(lambda s: s, [history_state], chatbot)

        reset_btn.click(
            clear_fn,
            outputs=[
                chatbot,
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio
            ]
        )
        input_mode.input(
            clear_fn,
            outputs=[
                chatbot,
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio]).then(
            update_input_interface,
            inputs=[input_mode],
            outputs=[
                audio,
                text_input
            ]
        )

    initialize_fn()
    # Launch the interface
    demo.launch(
        server_port=args.port,
        server_name=args.host
    )
