import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import GenerationConfig, TopKLogitsWarper, LogitsProcessorList

from src.caption.internvl.InternVL.internvl_chat.internvl.train.constants import (
    
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    
)
IMAGE_PLACEHOLDER = '<image>'
from src.caption.internvl.InternVL.internvl_chat.internvl.conversation import get_conv_template, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Константы для нормализации изображений (из InternVL)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from PIL import Image
import requests
from io import BytesIO
import re
import numpy as np
from methods.utils import load_images, string_to_token_ids

from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList



def get_vocab_embeddings_internvl(model, tokenizer, device="cuda"):
    """
    Get the token embeddings from InternVL2_5-1B model.

    Args:
        model: The InternVLChatModel instance (e.g., loaded from InternVL2_5-1B).
        tokenizer: The tokenizer compatible with the model (e.g., AutoTokenizer).
        device: The device to place the token IDs tensor on (default: "cuda").

    Returns:
        torch.Tensor: The embeddings for all tokens in the vocabulary.
    """
    vocab = tokenizer.get_vocab()
    token_ids = torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    token_embeddings = model.get_input_embeddings()(token_ids)
    return token_embeddings



def generate_text_prompt(model, model_name, text_prompt, num_patches=1):
    """
    Weaves in the image token placeholders into the provided text prompt for InternVL2_5-1B.

    Args:
        model: The InternVLChatModel instance (e.g., loaded from InternVL2_5-1B).
        model_name: Name of the model (e.g., "OpenGVLab/InternVL2_5-1B").
        text_prompt: The input text prompt, potentially containing IMAGE_PLACEHOLDER.
        num_patches: Number of image patches (default: 1).

    Returns:
        str: The formatted prompt with image tokens, ready for tokenization.
    """
    qs = text_prompt

    # Формируем строку токенов изображения
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN

    # Заменяем IMAGE_PLACEHOLDER на токены изображения или добавляем их
    if IMAGE_PLACEHOLDER in qs:
        qs = qs.replace(IMAGE_PLACEHOLDER, image_tokens, 1)
    else:
        qs = f"{image_tokens}\n{qs}"

    # Получаем шаблон разговора из модели
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], qs)
    template.append_message(template.roles[1], None)

    # Формируем итоговый запрос
    prompt = template.get_prompt()
    return prompt




def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_images(img_path):
    """
    Load images from a list of file paths.
    
    Args:
        image_files: List of paths to image files.
    
    Returns:
        List of PIL.Image objects.
    """
    return Image.open(img_path).convert('RGB')

def generate_images_tensor(model, img_path, image_processor=None):
    """
    Generate a tensor of images for InternVL2_5-1B.
    
    Args:
        model: The InternVLChatModel instance.
        img_path: Path to the image file (str).
        image_processor: Optional image processor (not used, kept for compatibility).
    
    Returns:
        tuple: (images_tensor, images, image_sizes)
            - images_tensor: Processed image tensor on the model's device.
            - images: List of PIL.Image objects.
            - image_sizes: List of original image sizes (width, height).
    """
    # Загрузка изображений
    image_files = img_path
    image = load_images(image_files)
    image_sizes = 224 

    # Получение размера изображения из конфигурации модели
    image_size = model.config.vision_config.image_size
    print('model.config.vision_config.image_size ', image_size)
    print('build_transform(image_sizes) ', image_sizes)
    # Создание трансформации для обработки изображений
    transform = build_transform(image_size)

    # Обработка изображений
    images_tensor = transform(image).unsqueeze(0).to(torch.float16).cuda()

    return images_tensor, image, image_sizes




def prompt_to_img_input_ids(prompt, tokenizer, device="cuda"):
    """
    Convert a prompt with image placeholders to input IDs for InternVL2_5-1B.

    Args:
        prompt: The input text prompt containing image tokens (e.g., <img><IMG_CONTEXT>...</img>).
        tokenizer: The tokenizer compatible with the model (e.g., AutoTokenizer).
        device: The device to place the input IDs tensor on (default: "cuda").

    Returns:
        torch.Tensor: Input IDs tensor with shape [1, seq_length] on the specified device.
    """
    # Токенизация промпта
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    return input_ids




def run_internvl_model(
    model,
    model_name,
    pixel_values,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
    num_patches=1
):
    """
    Run the InternVL2_5-1B model to generate text based on an image and text prompt.

    Args:
        model: The InternVLChatModel instance.
        model_name: Name of the model (e.g., "OpenGVLab/InternVL2_5-1B").
        pixel_values: Tensor of processed images (from generate_images_tensor).
        image_sizes: List of original image sizes (width, height).
        tokenizer: The tokenizer compatible with the model (e.g., AutoTokenizer).
        text_prompt: The input text prompt (default: "Write a detailed description.").
        hidden_states: Whether to return hidden states (default: False).
        num_patches: Number of image patches (default: 1).

    Returns:
        str or tuple: Decoded text output or (input_ids, output) if hidden_states=True.
    """
    if text_prompt is None:
        text_prompt = "Write a detailed description."

    # Формируем промпт с токенами изображения
    prompt = generate_text_prompt(model, model_name, text_prompt, num_patches=num_patches)
    
    # Токенизация промпта
    input_ids = prompt_to_img_input_ids(prompt, tokenizer, device=model.device)
    
    # Получаем шаблон разговора для определения стоп-токена
    template = get_conv_template(model.template)
    stop_str = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(stop_str)

    # Настраиваем параметры генерации
    generation_config = GenerationConfig(
        temperature=1.0,
        num_beams=5,
        max_new_tokens=512,
        eos_token_id=eos_token_id,
        use_cache=True
    )

    # Получаем attention_mask из токенизатора
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    with torch.inference_mode():
        output = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=hidden_states,
            return_dict_in_generate=True
        )

    if hidden_states:
        return input_ids, output

    # Декодируем выходные последовательности
    outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    # Удаляем стоп-строку и обрезаем пробелы
    outputs = outputs.split(stop_str)[0].strip()

    return outputs


def retrieve_logit_lens_internvl(state, img_path, text_prompt=None, num_patches=1):
    """
    Retrieve caption and softmax probabilities for image tokens from InternVL2_5-1B.

    Args:
        state: Dictionary containing model, model_name, tokenizer, and optional image_processor.
        img_path: Path to the image file (str).
        text_prompt: Input text prompt (default: None, uses "Write a detailed description.").
        num_patches: Number of image patches (default: 1).

    Returns:
        tuple: (caption, softmax_probs)
            - caption: Decoded text output (str).
            - softmax_probs: Softmax probabilities for image tokens, shape (vocab_dim, num_layers, num_tokens).
    """
    # Подготовка изображений
    pixel_values, images, image_sizes = generate_images_tensor(state["model"], img_path, image_processor=None)

    # Генерация выходных данных модели с hidden_states=True
    input_ids, output = run_internvl_model(
        state["model"],
        state["model_name"],
        pixel_values,
        image_sizes,
        state["tokenizer"],
        text_prompt=text_prompt,
        hidden_states=True,
        num_patches=num_patches
    )

    # Декодирование выходных последовательностей
    output_ids = output.sequences
    caption = state["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Обработка скрытых состояний
    hidden_states = torch.stack(output.hidden_states[0][-5:])
    print(hidden_states.shape)
    # Обработка логитов
    logits_warper = TopKLogitsWarper(top_k=50, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])
    
    with torch.inference_mode():
        curr_layer_logits = state["model"].get_output_embeddings()(hidden_states).to(torch.float16).cpu().float()
        print('curr_layer_logits ', curr_layer_logits)
        logit_scores = torch.nn.functional.log_softmax(curr_layer_logits, dim=-1)
        print('logit_scores ', logit_scores)
        logit_scores_processed = logits_processor(input_ids, logit_scores)
        print('logit_scores_processed ', logit_scores_processed)
        # logit_scores = logits_warper(input_ids, logit_scores_processed)
        print('logit_scores 2 ', logit_scores)
        softmax_probs = torch.nn.functional.softmax(logit_scores, dim=-1)
        print('softmax_probs ', softmax_probs)

    softmax_probs = softmax_probs.detach().cpu().numpy()

    # Находим индекс токена <IMG_CONTEXT> для выделения токенов изображения
    img_context_token_id = state["tokenizer"].convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    input_ids_list = input_ids.tolist()[0]
    try:
        image_token_index = input_ids_list.index(img_context_token_id)
    except ValueError:
        raise ValueError(f"Token {IMG_CONTEXT_TOKEN} not found in input_ids.")

    # Выделяем вероятности для токенов изображения
    num_image_tokens = state["model"].num_image_token * num_patches
    softmax_probs = softmax_probs[
        :, :, image_token_index:image_token_index + num_image_tokens
    ]

    # Транспонируем к форме (vocab_dim, num_layers, num_tokens)
    print("softmax_probs shape:", softmax_probs.shape)
    softmax_probs = softmax_probs.transpose(3, 0, 2, 1)

    return caption, softmax_probs


def reshape_internvl_prompt_hidden_layers(hidden_states):
    """
    Reshape hidden states of the prompt for InternVL2_5-1B to (num_layers, num_prompt_tokens, hidden_size).

    Args:
        hidden_states: Tensor of hidden states with shape (num_layers, batch_size, seq_length, hidden_size).

    Returns:
        torch.Tensor: Reshaped hidden states with shape (num_layers, num_prompt_tokens, hidden_size).
    """
    # Проверяем, что hidden_states имеет ожидаемую форму
    if len(hidden_states.shape) != 4:
        raise ValueError(f"Expected hidden_states with 4 dimensions, got shape {hidden_states.shape}")

    # Убираем размерность batch_size (batch_size=1)
    prompt_hidden_states = hidden_states.squeeze(1)  # (num_layers, seq_length, hidden_size)

    return prompt_hidden_states


def get_hidden_text_embedding_internvl(
    target_word, model, vocab_embeddings, tokenizer, layer=5, device="cuda"
):
    """
    Get the hidden state embedding for a target word using InternVL2_5-1B.

    Args:
        target_word: The target word to tokenize and embed (str).
        model: The InternVLChatModel instance.
        vocab_embeddings: Tensor of vocabulary embeddings (from get_vocab_embeddings_internvl).
        tokenizer: The tokenizer compatible with the model (e.g., AutoTokenizer).
        layer: The model layer to extract the hidden state from (default: 5).
        device: The device to place the input IDs tensor on (default: "cuda").

    Returns:
        torch.Tensor: Hidden state embedding for the last token of the target word, shape (1, hidden_size).
    """
    # Токенизация целевого слова
    token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    input_ids = torch.tensor([token_ids]).to(device)  # (1, num_tokens)

    # Получаем шаблон разговора для определения стоп-токена
    template = get_conv_template(model.template)
    stop_str = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(stop_str)

    # Настраиваем параметры генерации
    generation_config = GenerationConfig(
        temperature=1.0,
        num_beams=5,
        max_new_tokens=10,
        eos_token_id=eos_token_id,
        use_cache=False
    )

    # Получаем attention_mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=None,  # Нет изображений
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # Преобразуем скрытые состояния
    hidden_states = torch.stack([hs[-1] for hs in output.hidden_states])  # (num_layers, 1, seq_length, hidden_size)
    hidden_states = reshape_internvl_prompt_hidden_layers(hidden_states)  # (num_layers, seq_length, hidden_size)

    # Проверка валидации
    last_token_id = token_ids[-1]
    dist = torch.norm(
        hidden_states[0, len(token_ids) - 1] - vocab_embeddings[0, last_token_id]
    )
    if dist > 0.1:
        print(
            f"Validation check failed: caption word {target_word} didn't match: {dist}"
        )

    # Возвращаем скрытое состояние для указанного слоя и последнего токена
    return hidden_states[layer, len(token_ids) - 1].unsqueeze(0)  # (1, hidden_size)


def get_caption_from_internvl(
    img_path, model, model_name, tokenizer, image_processor=None, text_prompt=None, num_patches=1
):
    """
    Generate a caption for an image using InternVL2_5-1B.

    Args:
        img_path: Path to the image file (str).
        model: The InternVLChatModel instance.
        model_name: Name of the model (e.g., "OpenGVLab/InternVL2_5-1B").
        tokenizer: The tokenizer compatible with the model (e.g., AutoTokenizer).
        image_processor: Optional image processor (not used, kept for compatibility).
        text_prompt: Optional input text prompt (default: None, uses "Write a detailed description.").
        num_patches: Number of image patches (default: 1).

    Returns:
        str: Generated caption for the image.
    """
    # Подготовка изображений
    pixel_values, images, image_sizes = generate_images_tensor(
        model, img_path, image_processor=None
    )

    # Генерация подписи
    new_caption = run_internvl_model(
        model,
        model_name,
        pixel_values,
        image_sizes,
        tokenizer,
        text_prompt=text_prompt,
        hidden_states=False,
        num_patches=num_patches
    )

    return new_caption


from transformers import AutoModel, AutoTokenizer


def load_internvl_state(device="cuda"):
    """
    Load the state for InternVL2_5-1B model, including model, tokenizer, and helper functions.

    Args:
        device: The device to place the model and tensors on (default: "cuda").

    Returns:
        dict: State containing model, tokenizer, vocabulary, embeddings, and helper functions.
    """
    # Загрузка модели и токенизатора
    model_path = "OpenGVLab/InternVL2_5-1B"
    model_name = model_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    # Получение словаря и эмбеддингов
    vocabulary = tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_internvl(model, tokenizer, device=device)

    # Вспомогательные функции
    execute_model = lambda img_path, text_prompt=None, image_embeddings=None: get_caption_from_internvl(
        img_path, model, model_name, tokenizer, image_processor=None, text_prompt=text_prompt, num_patches=1
    )
    register_hook = (
        lambda hook, layer: model.language_model.model.layers[layer].register_forward_hook(hook)
    )
    register_pre_hook = (
        lambda pre_hook, layer: model.language_model.model.layers[layer].register_forward_pre_hook(pre_hook)
    )
    hidden_layer_embedding = lambda text, layer: get_hidden_text_embedding_internvl(
        text, model, vocab_embeddings, tokenizer, layer, device=device
    )

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "tokenizer": tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": model,
        "model_name": model_name,
        "image_processor": None,
    }
