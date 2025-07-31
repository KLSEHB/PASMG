import shutup; shutup.please()
import json
import argparse
import torch
import logging
import random
import numpy as np
import os
from transformers import get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler

from model1 import Model_1
from model2 import Model_2
from model3 import Model_3
from model4 import Model_4
from model5 import Model_5
from model6 import Model_6

from sklearn.metrics import precision_recall_curve, recall_score,precision_score,f1_score,confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
logger = logging.getLogger(__name__)

# class InputFeatures(object):
#     """A single training/test features for a example."""
#     def __init__(self, path1_ids, path2_ids, path3_ids, label):
#         self.path1_ids = path1_ids
#         self.path2_ids = path2_ids
#         self.path3_ids = path3_ids
#         self.label = label

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, path_ids_list, label):
        self.path_ids_list = path_ids_list
        self.label = label

def convert_code_to_token_ids(js, tokenizer, args):
    path_ids_list = []

    for i in range(1, args.PathNum + 1):
        path_key = f'path{i}'
        if path_key not in js:
            raise KeyError(f"The sample is missing the essential key: '{path_key}'")

        code_tokens = tokenizer.tokenize(js[path_key])[:args.block_size - 2]
        # print(js[path_key])
        # print(code_tokens)
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding = [tokenizer.pad_token_id] * (args.block_size - len(source_ids))
        path_ids_list.append(source_ids + padding)

    label = js.get("target")
    if label is None:
        raise KeyError("The sample is missing the 'target' key")

    return InputFeatures(
        path_ids_list=path_ids_list,
        label=label
    )


class PathsDataset(Dataset):
    def __init__(self, tokenizer, args, dataset_type):
        if dataset_type == "train":
            dataset_path = args.ExecutionPaths_train_set
        elif dataset_type == "eval":
            dataset_path = args.ExecutionPaths_valid_set
        elif dataset_type == "test":
            dataset_path = args.ExecutionPaths_test_set
        else:
            raise ValueError("Unknown dataset_type: {}".format(dataset_type))
        self.examples = []

        with open(dataset_path) as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            for line in tqdm(f, total=total_lines, desc=f"Tokenizing {dataset_type} dataset"):
                js = json.loads(line.strip())
                self.examples.append(convert_code_to_token_ids(js, tokenizer, args))

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        example = self.examples[i]
        path_tensors = [torch.tensor(path_ids) for path_ids in example.path_ids_list]
        return (*path_tensors, torch.tensor(example.label))

def train(args, train_dataset, eval_dataset, model):
    """ Train the model """
    # build dataloader
    logger.info("***** build dataloader *****")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    logger.info("***** evaluate the model per epoch *****")
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    logger.info("***** Prepare optimizer and schedule (linear warmup and decay) *****")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training

    if args.n_gpu > 1:
        logger.info("***** multi-gpu training *****")
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        tr_num = 0
        train_loss = 0
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(bar):
            paths = [tensor.to(args.device) for tensor in batch[:-1]]
            labels = batch[-1].to(args.device)
            model.train()
            model_inputs = {
                f'input_ids{i + 1}': path
                for i, path in enumerate(paths)
            }
            model_inputs["labels"] = labels
            loss = model(**model_inputs)
            if args.n_gpu > 1:
                loss = loss.mean()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} avg_loss {} loss {:.4f}".format(idx, avg_loss, loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        f1_at_best_threshold = evaluate(args, eval_dataset, model)
        if f1_at_best_threshold > best_f1:

            best_f1 = f1_at_best_threshold


            # checkpoint_prefix = 'checkpoint-best-f1'
            # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_file_path = os.path.join(args.output_dir, '{}'.format("model.bin"))
            torch.save(model_to_save.state_dict(),  model_file_path)

            threshold_file_path = os.path.join(args.output_dir, '{}'.format("threshold.json"))
            data = {
                "Best threshold": float(args.best_threshold)
            }
            with open(threshold_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logger.info("  " + "*" * 20)
            logger.info("  Best f1:%s", round(best_f1, 4))
            logger.info("  Best threshold:%s", round(args.best_threshold, 4))
            logger.info("  Saving model checkpoint to %s", model_file_path)
            logger.info("  Saving best threshold to %s", threshold_file_path)
            logger.info("  " + "*" * 20)


def evaluate(args, eval_dataset, model):

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # # model.to(args.device)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # logger.info("***** Eval results at the best threshold *****")


    model.eval()
    probs = []
    y_trues = []
    for batch in tqdm(eval_dataloader, desc="Making Predictions"):
        paths = [tensor.to(args.device) for tensor in batch[:-1]]
        labels = batch[-1].to(args.device)
        model.train()
        model_inputs = {
            f'input_ids{i + 1}': path
            for i, path in enumerate(paths)
        }
        with torch.no_grad():
            prob = model(**model_inputs)
            probs.append(prob.cpu().numpy().reshape(-1))
            y_trues.append(labels.cpu().numpy())

    probs = np.concatenate(probs, 0)
    y_trues = np.concatenate(y_trues, 0)

    args.best_threshold = calculate_best_f1_threshold(y_trues, probs)
    # args.best_threshold = calculate_balanced_f1_threshold(y_trues, probs)
    # best_threshold = 0.5

    y_preds = (probs > args.best_threshold).astype(int)  # 直接使用概率值

    # print_top_n_f1_thresholds(y_trues, probs, n=10)

    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)


    result = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "TNR": tnr,
        "eval_acc": acc,
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "threshold":args.best_threshold,
    }

    # PrecisionRecallDisplay.from_predictions(y_trues, probs, name='LineVul')
    # plt.savefig(f'eval_precision_recall_{args.model_name}.pdf')

    logger.info("***** Eval results at the best threshold *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return f1

def test(args, test_dataset, model):

    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # # model.to(args.device)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # logger.info("***** Eval results at the best threshold *****")


    model.eval()
    probs = []
    y_trues = []
    for batch in tqdm(test_dataloader, desc="Making Predictions"):
        paths = [tensor.to(args.device) for tensor in batch[:-1]]
        labels = batch[-1].to(args.device)
        model.train()
        model_inputs = {
            f'input_ids{i + 1}': path
            for i, path in enumerate(paths)
        }
        with torch.no_grad():
            prob = model(**model_inputs)
            probs.append(prob.cpu().numpy().reshape(-1))  # 处理形状
            y_trues.append(labels.cpu().numpy())

    probs = np.concatenate(probs, 0)
    y_trues = np.concatenate(y_trues, 0)

    if args.threshold is not None:
        threshold = args.threshold

    else:
        # checkpoint_prefix = 'checkpoint-best-f1'
        # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        threshold_file_path = os.path.join(args.output_dir, '{}'.format("threshold.json"))
        with open(threshold_file_path, 'r') as f:
            threshold = json.load(f)
        args.best_threshold = threshold['Best threshold']
        threshold = args.best_threshold

    y_preds = (probs > threshold).astype(int)

    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    result = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "TNR": tnr,
        "test_acc": acc,
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "threshold":threshold,
    }

    logger.info("***** Test results at the best threshold *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    if args.write_results:
        with open(args.write_results_path, 'a', encoding='utf-8') as file:
            file.write('TP: {}, TN: {}, FP: {}, FN: {}\n'.format(tp, tn, fp, fn))
            file.write('TPR: {}, TNR: {}\n'.format(tpr, tnr))
            file.write('ACC:       {}\n'.format(acc))
            file.write('Recall:    {}\n'.format(recall))
            file.write('Precision: {}\n'.format(precision))
            file.write('F1 Score:  {}\n'.format(f1))
            file.write('threshold: {}\n'.format(threshold))

def print_top_n_f1_thresholds(y_true, y_scores, n=10):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1_scores = []

    for i in tqdm(range(len(thresholds)), desc="Calculating F1 scores", unit="threshold"):

        y_pred = (y_scores > thresholds[i]).astype(int)

        current_f1 = f1_score(y_true, y_pred)

        f1_scores.append((current_f1, thresholds[i]))

    f1_scores.sort(reverse=True, key=lambda x: x[0])

    top_n_f1_scores = f1_scores[:n]
    for rank, (f1, threshold) in enumerate(top_n_f1_scores, start=1):
        print(f"Rank {rank}: F1 Score = {f1:.4f}, Threshold = {threshold:.4f}")

def calculate_best_f1_threshold(y_true, y_scores):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    best_f1 = -1
    best_threshold = 0


    # for i in range(len(thresholds)):
    for i in tqdm(range(len(thresholds)), desc="Finding best_threshold", unit="threshold"):

        y_pred = (y_scores > thresholds[i]).astype(int)


        current_f1 = f1_score(y_true, y_pred)


        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresholds[i]

        # print(f"Threshold: {thresholds[i]:.4f}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {current_f1:.4f}")

    # print(f"\nBest F1 Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")
    return best_threshold

def calculate_balanced_f1_threshold(y_true, y_scores):
    alpha = 0.2
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    best_score = -1
    best_threshold = 0

    for i in tqdm(range(len(thresholds)), desc="Finding balanced threshold", unit="threshold"):
        y_pred = (y_scores > thresholds[i]).astype(int)
        p = precision[i]
        r = recall[i]
        f1 = f1_score(y_true, y_pred)

        penalty = alpha * abs(p - r)
        score = f1 - penalty

        if score > best_score:
            best_score = score
            best_threshold = thresholds[i]

    return best_threshold

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    # print("Config file content:")
    # print(json.dumps(config, indent=4))

    parser = argparse.ArgumentParser()
    parser.add_argument('--ExecutionPaths_train_set', default=config['ExecutionPaths_train_set'], type=str)
    parser.add_argument('--ExecutionPaths_valid_set', default=config['ExecutionPaths_valid_set'], type=str)
    parser.add_argument('--ExecutionPaths_test_set', default=config['ExecutionPaths_test_set'], type=str)
    parser.add_argument('--pretrained_model_path', default=config['pretrained_model_path'], type=str)
    parser.add_argument('--output_dir', default=config['output_dir'], type=str)
    parser.add_argument('--write_results_path', type=str)

    parser.add_argument('--block_size', default=config['block_size'], type=int)
    parser.add_argument('--train_batch_size', default=config['train_batch_size'], type=int)
    parser.add_argument('--eval_batch_size', default=config['eval_batch_size'], type=int)
    parser.add_argument('--epochs', default=config['epochs'], type=int)
    parser.add_argument('--gradient_accumulation_steps', default=config['gradient_accumulation_steps'], type=int)
    parser.add_argument('--seed', default=config['seed'], type=int)
    parser.add_argument("--PathNum", default=config['PathNum'], type=int)

    parser.add_argument('--weight_decay', default=config['weight_decay'], type=float)
    parser.add_argument('--learning_rate', default=config['learning_rate'], type=float)
    parser.add_argument('--adam_epsilon', default=config['adam_epsilon'], type=float)
    parser.add_argument('--max_grad_norm', default=config['max_grad_norm'], type=float)
    parser.add_argument('--threshold', default=config['threshold'], type=float)

    parser.add_argument('--do_train', default=config['do_train'])
    parser.add_argument('--do_test', default=config['do_test'])
    parser.add_argument('--write_results', default=config['write_results'])

    args = parser.parse_args()

    print("=============== Arguments ===============")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print ("========================================")

    if args.write_results:
        with open(args.write_results_path, 'a', encoding='utf-8') as file:
            # 向文件中追加写入内容
            file.write('========================================\n')
            file.write('ExecutionPaths_train_set: {}\n'.format(args.ExecutionPaths_train_set))
            file.write('ExecutionPaths_valid_set: {}\n'.format(args.ExecutionPaths_valid_set))
            file.write('ExecutionPaths_test_set: {}\n'.format(args.ExecutionPaths_test_set))

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu, )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    pre_trained_model_config = RobertaConfig.from_pretrained(args.pretrained_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)
    pre_trained_model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path, config=pre_trained_model_config,
                                                             ignore_mismatched_sizes=True)

    # if args.use_model == "model_CLS":
    #     model = Model_CLS(pre_trained_model, pre_trained_model_config, args)
    # elif args.use_model == "model_CLS_AND_TextCNN":
    #     model = Model_CLS_AND_TextCNN(pre_trained_model, pre_trained_model_config, args)
    # elif args.use_model == "model_CLS_AND_BLSTM":
    #     model = Model_CLS_AND_BLSTM(pre_trained_model, pre_trained_model_config, args)
    # else:
    #     raise ValueError("Unknown model type: {}".format(args.use_model))

    if args.PathNum == 1:
        model = Model_1(pre_trained_model, pre_trained_model_config, args)
    elif args.PathNum == 2:
        model = Model_2(pre_trained_model, pre_trained_model_config, args)
    elif args.PathNum == 3:
        model = Model_3(pre_trained_model, pre_trained_model_config, args)
    elif args.PathNum == 4:
        model = Model_4(pre_trained_model, pre_trained_model_config, args)
    elif args.PathNum == 5:
        model = Model_5(pre_trained_model, pre_trained_model_config, args)
    elif args.PathNum == 6:
        model = Model_6(pre_trained_model, pre_trained_model_config, args)
    else:
        raise ValueError("Unknown model type: {}".format(args.use_model))

    if args.do_train:

        train_dataset = PathsDataset(tokenizer, args, "train")
        eval_dataset = PathsDataset(tokenizer, args, "eval")
        train(args, train_dataset, eval_dataset, model)

    if args.do_test:
        checkpoint_prefix = f'model.bin'
        saved_model_path = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(saved_model_path, map_location=args.device), strict=False)
        model.to(args.device)
        test_dataset = PathsDataset(tokenizer, args, "test")
        test(args, test_dataset, model)




if __name__ == "__main__":
    main()