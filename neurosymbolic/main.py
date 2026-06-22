import json
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural.pathways2nl.llm import LocalLLM


def main(argv):
    if len(argv) > 1:
        with open(argv[1]) as f:
            conf = json.load(f)
    else:
        raise ValueError("Config file required")
    
    models = conf["models"]
    batch_size = conf["batch_size"]
    batch_size = 20
    dataset_dir = "datasets"
    dataset_files = [f for f in os.listdir(dataset_dir)]
    
    with open("neurosymbolic/prompt.json") as f:
        templates = json.load(f)
    
    for dataset_file in dataset_files:
        base_name = dataset_file.replace("_dataset.json", "")
        print(f"\nProcessing: {base_name}")
        
        with open(os.path.join(dataset_dir, dataset_file)) as f:
            problems = json.load(f)
        
        sentences = []
        for problem in problems:
            sentences.append(problem["P1"])
            sentences.append(problem["P2"])
            sentences.append(problem["C"])

        for model in tqdm(models, desc=f"Models for {base_name}"):
            print(f"  Model: {model}")
            
            llm = LocalLLM(model)
            
            #batch_size = 40 if "Mixtral" in model else batch_size
            
            parses = []
            
            for i in tqdm(range(0, len(sentences), batch_size)):
                batch = sentences[i:i+batch_size]
                
                
                responses = llm.prompt(
                    template=templates["CONTEXT"]+templates["TASK"],
                    question=batch,
                    examples=templates["EXAMPLES"],
                    gtd=[""] * len(batch),
                    output_size=512
                )
                parses.extend(responses)
                
  
            
            output_file = f"neurosymbolic/ccg/{model.replace('/', '--')}/{base_name}_llm_ccg.pl"
            os.makedirs(f"neurosymbolic/ccg/{model.replace('/', '--')}", exist_ok=True)
            
            with open(output_file, "w") as f:
                for idx, parse in enumerate(parses, start=1):
                    f.write(f"% sentence id: {idx}\n")
                    f.write(f"{parse}\n\n")
                
            
            print(f"    Saved: {output_file}")


if __name__ == "__main__":
    main(sys.argv)