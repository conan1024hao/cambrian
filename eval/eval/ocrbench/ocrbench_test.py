import os
import json
import csv
from datetime import datetime


current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def edit_distance(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        dp[i][0] = i
    for j in range(len(str2) + 1):
        dp[0][j] = j

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[len(str1)][len(str2)]


def get_accuracy(pred_list, output_file, write_bool, extra_penalization):
    correct = 0.0
    for items in pred_list:
        pred_answer = items["pred_answer"]
        gt_answers = [x for x in items["gt_answers"]]
        found = False
        if extra_penalization:
            for x in gt_answers:
                if edit_distance(pred_answer, x) == 0:
                    found = True
                    break
        else:
            for x in gt_answers:
                if x in pred_answer:
                    found = True
                    break
        if found:
            correct += 1.0
        elif write_bool:
            output_file = os.path.expanduser(output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'a') as out_file:
                out_file.write(json.dumps(items) + '\n')
    return correct/len(pred_list)


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None, extra_penalization=True):
    pred_list = {}
    full_list = []
    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            model = data.get('model_id', '')
            category = data.get('category', '')

            if category == "Handwritten Mathematical Expression Recognition":
                answer = data.get('answer', '').strip().replace('\n', ' ').replace(' ', '')
                gt_answer = [x.strip().replace('\n', ' ').replace(' ', '') for x in data.get('gt_answer', [''])]
            else:
                answer = data.get('answer', '').lower().strip().replace('\n', ' ')
                gt_answer = [x.lower().strip().replace('\n', ' ') for x in data.get('gt_answer', [''])]

            full_list.append({
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                })
            if category in pred_list:
                pred_list[category].append({
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                })
            else:
                pred_list[category] = [{
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                }]

    combined_data = {
        "model": model,
        "time": time_string,
    }
    for category, preds in pred_list.items():
        combined_data[category] = {'accuracy': 100.0 * get_accuracy(preds, output_file, False, extra_penalization), 'total': len(preds)}

    combined_data['total_accuracy'] = {'accuracy': 100.0 * get_accuracy(full_list, output_file, True, extra_penalization), 'total': len(full_list)}
    add_data_to_csv(csv_file, combined_data)

    print(f"Model: {model}")
    print(f"Saved incorrect predictions to {output_file}")
    print(f"Combined data: {combined_data}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"ocrbench_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Saved extra experiment data to {extra_csv_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl", help="Path to the jsonl file containing the model predictions")
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to the output file to store the incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")
    parser.add_argument("--extra_penalization", action='store_true', help="To not penalize if the predicted answers contains extra information before or after the desired prediction")
    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir, args.extra_penalization)
