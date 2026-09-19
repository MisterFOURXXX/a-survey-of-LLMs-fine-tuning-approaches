import torch


def generate_predictions(model, tokenizer, prompts, max_new_tokens=64):
    model.eval()
    predictions = []

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        predictions.append(text)

    return predictions