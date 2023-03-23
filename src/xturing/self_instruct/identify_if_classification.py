import json
import os
import random
from collections import OrderedDict

from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1
from tqdm import tqdm

random.seed(42)


templates = {"template_1": template_1}


# Creates a function out of the main function that accepts batch_dir, num_instructions, template, engine, request_batch_size, api_key, and organization as arguments.


def identify_tasks_as_type_classification(
    target_dir,
    num_instructions,
    template,
    engine,
    request_batch_size,
    api_key,
    organization,
):
    # Load the machine generated instructions
    with open(os.path.join(target_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if num_instructions is not None:
            lines = lines[:num_instructions]

    # Create the output path
    output_path = os.path.join(target_dir, f"is_clf_or_not_{engine}_{template}.jsonl")

    # Load the existing requests
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    # Create the progress bar
    progress_bar = tqdm(total=len(lines))

    # Write the output to file
    with open(output_path, "w") as fout:
        # Iterate over the lines in batches
        for batch_idx in range(0, len(lines), request_batch_size):
            batch = [
                json.loads(line)
                for line in lines[batch_idx : batch_idx + request_batch_size]
            ]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[template]
                prompts = [
                    prefix
                    + " "
                    + d["instruction"].strip()
                    + "\n"
                    + "Is it classification?"
                    for d in batch
                ]
                results = make_gpt3_requests(
                    engine=engine,
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=api_key,
                    organization=organization,
                )
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][
                            0
                        ]["text"]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"],
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
