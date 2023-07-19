"""Run codes."""
# pylint: disable=line-too-long, broad-exception-caught, invalid-name, missing-function-docstring, too-many-instance-attributes, missing-class-docstring
# ruff: noqa: E501
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import gradio as gr
import psutil
from about_time import about_time

# from ctransformers import AutoConfig, AutoModelForCausalLM
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from loguru import logger

filename_list = [
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q2_K.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q3_K_L.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q3_K_M.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q3_K_S.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_1.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_K_M.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_K_S.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q5_0.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q5_1.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q5_K_M.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q5_K_S.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q6_K.bin",
    "Wizard-Vicuna-7B-Uncensored.ggmlv3.q8_0.bin",
]

URL = "https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/raw/main/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_K_M.bin"  # 4.05G
MODEL_FILENAME = Path(URL).name
MODEL_FILENAME = filename_list[0]  # q2_K 4.05G
MODEL_FILENAME = filename_list[5]  # q4_1 4.21

REPO_ID = "/".join(
    urlparse(URL).path.strip("/").split("/")[:2]
)  # TheBloke/Wizard-Vicuna-7B-Uncensored-GGML

DESTINATION_FOLDER = "models"

os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

ns = SimpleNamespace(
    response="",
    generator=[],
)

default_system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."

user_prefix = "[user]: "
assistant_prefix = "[assistant]: "


def predict_str(prompt, bot):  # bot is in fact bot_history
    # logger.debug(f"{prompt=}, {bot=}, {timeout=}")

    if bot is None:
        bot = []

    logger.debug(f"{prompt=}, {bot=}")

    try:
        # user_prompt = prompt
        generator = generate(
            LLM,
            GENERATION_CONFIG,
            system_prompt=default_system_prompt,
            user_prompt=prompt.strip(),
        )

        ns.generator = generator  # for .then

    except Exception as exc:
        logger.error(exc)

    # bot.append([prompt, f"{response} {_}"])
    # return prompt, bot

    _ = bot + [[prompt, None]]
    logger.debug(f"{prompt=}, {_=}")

    return prompt, _


def bot_str(bot):
    if bot:
        bot[-1][1] = ""
    else:
        bot = [["Something is wrong", ""]]

    print(assistant_prefix, end=" ", flush=True)

    response = ""

    flag = 1
    then = time.time()
    for word in ns.generator:
        # record first response time
        if flag:
            logger.debug(f"\t {time.time() - then:.1f}s")
            flag = 0
        print(word, end="", flush=True)
        # print(word, flush=True)  # vertical stream
        response += word
        bot[-1][1] = response
        yield bot


def predict(prompt, bot):
    # logger.debug(f"{prompt=}, {bot=}, {timeout=}")
    logger.debug(f"{prompt=}, {bot=}")

    ns.response = ""
    then = time.time()
    with about_time() as atime:  # type: ignore
        try:
            # user_prompt = prompt
            generator = generate(
                LLM,
                GENERATION_CONFIG,
                system_prompt=default_system_prompt,
                user_prompt=prompt.strip(),
            )

            ns.generator = generator  # for .then

            print(assistant_prefix, end=" ", flush=True)

            response = ""
            buff.update(value="diggin...")

            flag = 1
            for word in generator:
                # record first response time
                if flag:
                    logger.debug(f"\t {time.time() - then:.1f}s")
                    flag = 0
                # print(word, end="", flush=True)
                print(word, flush=True)  # vertical stream
                response += word
                ns.response = response
                buff.update(value=response)
            print("")
            logger.debug(f"{response=}")
        except Exception as exc:
            logger.error(exc)
            response = f"{exc=}"

    # bot = {"inputs": [response]}
    _ = (
        f"(time elapsed: {atime.duration_human}, "  # type: ignore
        f"{atime.duration/(len(prompt) + len(response)):.1f}s/char)"  # type: ignore
    )

    bot.append([prompt, f"{response} {_}"])

    return prompt, bot


def predict_api(prompt):
    logger.debug(f"{prompt=}")
    ns.response = ""
    try:
        # user_prompt = prompt
        _ = GenerationConfig(
            temperature=0.2,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            max_new_tokens=512,  # adjust as needed
            seed=42,
            reset=False,  # reset history (cache)
            stream=True,  # TODO stream=False and generator
            threads=os.cpu_count() // 2,  # type: ignore  # adjust for your CPU
            stop=["<|im_end|>", "|<"],
        )

        # TODO: stream does not make sense in api?
        generator = generate(
            LLM, _, system_prompt=default_system_prompt, user_prompt=prompt.strip()
        )
        print(assistant_prefix, end=" ", flush=True)

        response = ""
        buff.update(value="diggin...")
        for word in generator:
            print(word, end="", flush=True)
            response += word
            ns.response = response
            buff.update(value=response)
        print("")
        logger.debug(f"{response=}")
    except Exception as exc:
        logger.error(exc)
        response = f"{exc=}"
    # bot = {"inputs": [response]}
    # bot = [(prompt, response)]

    return response


def download_quant(destination_folder: str, repo_id: str, model_filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=True,
    )


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(system_prompt: str, user_prompt: str):
    """Format prompt based on: https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py."""
    # TODO: fix prompts

    system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    assistant_prompt = "<|im_start|>assistant\n"

    return f"{system_prompt}{user_prompt}{assistant_prompt}"


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    system_prompt: str = default_system_prompt,
    user_prompt: str = "",
):
    """Run model inference, will return a Generator if streaming is true."""
    # if not user_prompt.strip():
    return llm(
        format_prompt(
            system_prompt,
            user_prompt,
        ),
        **asdict(generation_config),
    )


# if "mpt" in model_filename:
#     config = AutoConfig.from_pretrained("mosaicml/mpt-30b-cha t", context_length=8192)
#     llm = AutoModelForCausalLM.from_pretrained(
#         os.path.abspath(f"models/{model_filename}"),
#         model_type="mpt",
#         config=config,
#     )

# https://huggingface.co/spaces/matthoffner/wizardcoder-ggml/blob/main/main.py
_ = """
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/WizardCoder-15B-1.0-GGML",
    model_file="WizardCoder-15B-1.0.ggmlv3.q4_0.bin",
    model_type="starcoder",
    threads=8
)
# """

logger.info(f"start dl, {REPO_ID=}, {MODEL_FILENAME=}, {DESTINATION_FOLDER=}")
download_quant(DESTINATION_FOLDER, REPO_ID, MODEL_FILENAME)
logger.info("done dl")

logger.debug(f"{os.cpu_count()=} {psutil.cpu_count(logical=False)=}")
cpu_count = os.cpu_count() // 2  # type: ignore
cpu_count = psutil.cpu_count(logical=False)

logger.debug(f"{cpu_count=}")

logger.info("load llm")

_ = Path("models", MODEL_FILENAME).absolute().as_posix()
logger.debug(f"model_file: {_}, exists: {Path(_).exists()}")
LLM = AutoModelForCausalLM.from_pretrained(
    # "TheBloke/WizardCoder-15B-1.0-GGML",
    REPO_ID,  # DESTINATION_FOLDER,  # model_path_or_repo_id: str required
    model_file=_,
    model_type="llama",  # "starcoder",  AutoConfig.from_pretrained(REPO_ID)
    threads=cpu_count,
)

logger.info("done load llm")

GENERATION_CONFIG = GenerationConfig(
    temperature=0.2,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjust as needed
    seed=42,
    reset=False,  # reset history (cache)
    stream=True,  # streaming per word/token
    threads=cpu_count,
    stop=["<|im_end|>", "|<"],  # TODO possible fix of stop
)

css = """
    .importantButton {
        background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
        border: none !important;
    }
    .importantButton:hover {
        background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
        border: none !important;
    }
    .disclaimer {font-variant-caps: all-small-caps; font-size: xx-small;}
    .xsmall {font-size: x-small;}
"""
etext = """In America, where cars are an important part of the national psyche, a decade ago people had suddenly started to drive less, which had not happened since the oil shocks of the 1970s. """
examples = [
    ["How to pick a lock? Provide detailed steps."],
    ["Explain the plot of Cinderella in a sentence."],
    [
        "How long does it take to become proficient in French, and what are the best methods for retaining information?"
    ],
    ["What are some common mistakes to avoid when writing code?"],
    ["Build a prompt to generate a beautiful portrait of a horse"],
    ["Suggest four metaphors to describe the benefits of AI"],
    ["Write a pop song about leaving home for the sandy beaches."],
    ["Write a summary demonstrating my ability to tame lions"],
    ["鲁迅和周树人什么关系 说中文"],
    ["鲁迅和周树人什么关系"],
    ["鲁迅和周树人什么关系 用英文回答"],
    ["从前有一头牛，这头牛后面有什么？"],
    ["正无穷大加一大于正无穷大吗？"],
    ["正无穷大加正无穷大大于正无穷大吗？"],
    ["-2的平方根等于什么"],
    ["树上有5只鸟，猎人开枪打死了一只。树上还有几只鸟？"],
    ["树上有11只鸟，猎人开枪打死了一只。树上还有几只鸟？提示：需考虑鸟可能受惊吓飞走。"],
    ["以红楼梦的行文风格写一张委婉的请假条。不少于320字。"],
    [f"{etext} 翻成中文，列出3个版本"],
    [f"{etext} \n 翻成中文，保留原意，但使用文学性的语言。不要写解释。列出3个版本"],
    ["假定 1 + 2 = 4, 试求 7 + 8"],
    ["判断一个数是不是质数的 javascript 码"],
    ["实现python 里 range(10)的 javascript 码"],
    ["实现python 里 [*(range(10)]的 javascript 码"],
    ["Erkläre die Handlung von Cinderella in einem Satz."],
    ["Erkläre die Handlung von Cinderella in einem Satz. Auf Deutsch"],
]

with gr.Blocks(
    # title="mpt-30b-chat-ggml",
    title=f"{MODEL_FILENAME}",
    theme=gr.themes.Soft(text_size="sm", spacing_size="sm"),
    css=css,
) as block:
    with gr.Accordion("🎈 Info", open=False):
        # gr.HTML(
        #     """<center><a href="https://huggingface.co/spaces/mikeee/mpt-30b-chat?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate"></a> and spin a CPU UPGRADE to avoid the queue</center>"""
        # )
        gr.Markdown(
            f"""<h5><center><{REPO_ID}>{MODEL_FILENAME}</center></h4>
            The bot only speaks English.

            Most examples are meant for another model.
            You probably should try to test
            some related prompts.
            """,
            elem_classes="xsmall",
        )

    # chatbot = gr.Chatbot().style(height=700)  # 500
    chatbot = gr.Chatbot(height=500)
    buff = gr.Textbox(show_label=False, visible=False)
    with gr.Row():
        with gr.Column(scale=5):
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Ask me anything (press Enter or click Submit to send)",
                show_label=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            with gr.Row():
                submit = gr.Button("Submit", elem_classes="xsmall")
                stop = gr.Button("Stop", visible=False)
                clear = gr.Button("Clear History", visible=True)
    with gr.Row(visible=False):
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    system = gr.Textbox(
                        label="System Prompt",
                        value=default_system_prompt,
                        show_label=False,
                    ).style(container=False)
                with gr.Column():
                    with gr.Row():
                        change = gr.Button("Change System Prompt")
                        reset = gr.Button("Reset System Prompt")

    with gr.Accordion("Example Inputs", open=True):
        examples = gr.Examples(
            examples=examples,
            inputs=[msg],
            examples_per_page=40,
        )

    # with gr.Row():
    with gr.Accordion("Disclaimer", open=False):
        _ = "-".join(MODEL_FILENAME.split("-")[:2])
        gr.Markdown(
            f"Disclaimer: {_} can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. {_} was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    _ = """
    msg.submit(
        # fn=conversation.user_turn,
        fn=predict,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        # queue=True,
        show_progress="full",
        api_name="predict",
    )
    submit.click(
        fn=lambda x, y: ("",) + predict(x, y)[1:],  # clear msg
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
    )
    # """
    msg.submit(
        # fn=conversation.user_turn,
        fn=predict_str,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
        api_name="predict",
    ).then(bot_str, chatbot, chatbot)
    submit.click(
        fn=lambda x, y: ("",) + predict_str(x, y)[1:],  # clear msg
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
    ).then(bot_str, chatbot, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

    # update buff Textbox, every: units in seconds)
    # https://huggingface.co/spaces/julien-c/nvidia-smi/discussions
    # does not work
    # AttributeError: 'Blocks' object has no attribute 'run_forever'
    # block.run_forever(lambda: ns.response, None, [buff], every=1)

    with gr.Accordion("For Chat/Translation API", open=False, visible=False):
        input_text = gr.Text()
        api_btn = gr.Button("Go", variant="primary")
        out_text = gr.Text()
    api_btn.click(
        predict_api,
        input_text,
        out_text,
        # show_progress="full",
        api_name="api",
    )

# concurrency_count=5, max_size=20
# max_size=36, concurrency_count=14
block.queue(concurrency_count=5, max_size=20).launch(debug=True)
