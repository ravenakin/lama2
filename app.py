"""Run codes."""
# pylint: disable=line-too-long, broad-exception-caught, invalid-name, missing-function-docstring, too-many-instance-attributes, missing-class-docstring
# ruff: noqa: E501
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import gradio as gr
import psutil
from about_time import about_time

# from ctransformers import AutoConfig, AutoModelForCausalLM
from ctransformers import AutoModelForCausalLM

# from huggingface_hub import hf_hub_download
from dl_hf_model import dl_hf_model
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

url = "https://huggingface.co/savvamadar/ggml-gpt4all-j-v1.3-groovy/blob/main/ggml-gpt4all-j-v1.3-groovy.bin"
url = "https://huggingface.co/TheBloke/Llama-2-13B-GGML/blob/main/llama-2-13b.ggmlv3.q4_K_S.bin"  # 7.37G
# url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q3_K_L.bin"
url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q3_K_L.bin"  # 6.93G
# url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q3_K_L.binhttps://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q4_K_M.bin"  # 7.87G

url = "https://huggingface.co/localmodels/Llama-2-13B-Chat-ggml/blob/main/llama-2-13b-chat.ggmlv3.q4_K_S.bin" # 7.37G

prompt_template="""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: {user_prompt}

### Response:
"""

prompt_template = """System: You are a helpful,
respectful and honest assistant. Always answer as
helpfully as possible, while being safe.  Your answers
should not include any harmful, unethical, racist,
sexist, toxic, dangerous, or illegal content. Please
ensure that your responses are socially unbiased and
positive in nature. If a question does not make any
sense, or is not factually coherent, explain why instead
of answering something not correct. If you don't know
the answer to a question, please don't share false
information.
User: {prompt}
Assistant: """

prompt_template = """Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer."""

_ = [elm for elm in prompt_template.splitlines() if elm.strip()]
stop_string = [elm.split(":")[0] + ":" for elm in _][-2]

try:
    model_loc, file_size = dl_hf_model(url)
except Exception as exc_:
    logger.error(exc_)
    raise SystemExit(1) from exc_

logger.debug(f"{model_loc} {file_size}GB")

logger.debug(f"{stop_string=}")

_ = psutil.cpu_count(logical=False)
cpu_count: int = int(_) if _ else 1
logger.debug(f"{cpu_count=}")

logger.info("load llm")
_ = Path(model_loc).absolute().as_posix()
logger.debug(f"model_file: {_}, exists: {Path(_).exists()}")
LLM = None

if "okteto" in platform.node():
    # url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q2_K.bin"
    LLM = AutoModelForCausalLM.from_pretrained(
        "models/llama-2-13b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        threads=cpu_count,
    )
else:
    LLM = AutoModelForCausalLM.from_pretrained(
        model_loc,
        model_type="llama",
        threads=cpu_count,
    )

logger.info("done load llm")

os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

ns = SimpleNamespace(
    response="",
    generator=(_ for _ in []),
)


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    seed: int = 42
    reset: bool = False
    stream: bool = True
    threads: int = cpu_count
    # stop: list[str] = field(default_factory=lambda: [stop_string])


def generate(
    question: str,
    llm=LLM,
    generation_config: GenerationConfig = GenerationConfig(),
):
    """Run model inference, will return a Generator if streaming is true."""
    # _ = prompt_template.format(question=question)
    # print(_)
    return llm(
        question,
        **asdict(generation_config),
    )


logger.debug(f"{asdict(GenerationConfig())=}")


def predict_str(prompt, bot):  # bot is in fact bot_history
    # logger.debug(f"{prompt=}, {bot=}, {timeout=}")

    if bot is None:
        bot = []

    logger.debug(f"{prompt=}, {bot=}")

    try:
        # user_prompt = prompt
        generator = generate(
            prompt,
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
                prompt,
            )

            ns.generator = generator  # for .then

            print("--", end=" ", flush=True)

            response = ""
            buff.update(value="diggin...")

            flag = 1
            for word in generator:
                # record first response time
                if flag:
                    fisrt_arr = f"{time.time() - then:.1f}s"
                    logger.debug(f"\t 1st arrival: {fisrt_arr}")
                    flag = 0
                print(word, end="", flush=True)
                # print(word, flush=True)  # vertical stream
                response += word
                ns.response = f"({fisrt_arr}){response}"
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

    bot.append([prompt, f"{response} \n{_}"])

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
            stream=True,
            threads=cpu_count,
            # stop=prompt_prefix[1:2],
        )

        generator = generate(
            prompt,
        )

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


def update_buff():
    return ns.response


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
    ["What NFL team won the Super Bowl in the year Justin Bieber was born?"],
    ["What NFL team won the Super Bowl in the year Justin Bieber was born? Think step by step."],
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

logger.info("start block")

with gr.Blocks(
    title=f"{Path(model_loc).name}",
    theme=gr.themes.Soft(text_size="sm", spacing_size="sm"),
    css=css,
) as block:
    with gr.Accordion("🎈 Info", open=False):
        # gr.HTML(
        #     """<center><a href="https://huggingface.co/spaces/mikeee/mpt-30b-chat?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate"></a> and spin a CPU UPGRADE to avoid the queue</center>"""
        # )
        gr.Markdown(
            dedent(f"""<h5><center>{Path(model_loc).name}</center></h4>

            Most examples are meant for another model.
            You probably should try to test
            some related prompts. For example:\n
            {prompt_template}"""),
            elem_classes="xsmall",
        )

    # chatbot = gr.Chatbot().style(height=700)  # 500
    chatbot = gr.Chatbot(height=500)
    buff = gr.Textbox(show_label=False, visible=True)
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
                with gr.Column(scale=2, container=False):
                    system = gr.Textbox(
                        label="System Prompt",
                        value=prompt_template,
                        show_label=False,
                    # ).style(container=False)
                    )
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
        _ = Path(model_loc).name
        gr.Markdown(
            f"Disclaimer: {_} can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. {_} was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    # _ = """
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

    _ = """
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
    # """

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

    block.load(update_buff, [], buff, every=1)

# concurrency_count=5, max_size=20
# max_size=36, concurrency_count=14
# CPU cpu_count=2 16G, model 7G
# CPU UPGRADE cpu_count=8 32G, model 7G

_ = int(psutil.virtual_memory().total / 10**9 // file_size - 1)
concurrency_count = max(_, 1)
logger.info(f"{concurrency_count=}")

block.queue(concurrency_count=concurrency_count, max_size=5).launch(debug=True)
