import gradio as gr
from transformers import pipeline

token_skill_classifier = pipeline(model="jjzha/jobbert_skill_extraction", aggregation_strategy="first")
token_knowledge_classifier = pipeline(model="jjzha/jobbert_knowledge_extraction", aggregation_strategy="first")


examples = [
        "Knowing Python is a plus",
        "Recommend changes, develop and implement processes to ensure compliance with IFRS standards",
        "Experience with Unreal and/or Unity and/or native IOS/Android 3D development and/or Web based 3D engines",
        ]


def aggregate_span(results):
    new_results = []
    current_result = results[0]

    for result in results[1:]:
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = result

    new_results.append(current_result)

    return new_results

def ner(text):
    output_skills = token_skill_classifier(text)
    for result in output_skills:
        if result.get("entity_group"):
            result["entity"] = "Skill"
            del result["entity_group"]

    output_knowledge = token_knowledge_classifier(text)
    for result in output_knowledge:
        if result.get("entity_group"):
            result["entity"] = "Knowledge"
            del result["entity_group"]

    if len(output_skills) > 0:
        output_skills = aggregate_span(output_skills)
    if len(output_knowledge) > 0:
        output_knowledge = aggregate_span(output_knowledge)

    return {"text": text, "entities": output_skills}, {"text": text, "entities": output_knowledge}


demo = gr.Interface(fn=ner,
                    inputs=gr.Textbox(placeholder="Enter sentence here..."),
                    outputs=["highlight", "highlight"],
                    examples=examples)

demo.launch()