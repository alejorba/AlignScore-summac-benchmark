# TODO
# - Add default ckpt path for alignscore
# - Implement align_func for all metrics
# @ Implement score func

from argparse import ArgumentParser
import json
import os
import pickle
import time

import nltk
from nltk.tokenize import sent_tokenize
from summac.benchmark import SummaCBenchmark
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score

from baselines import *

class Timer():
    def __init__(self):
        self.start = time.time()
        self.save_path = './benchmarks/time.json'

    def finish(self, display_name):
        elapsed_time = time.time() - self.start
        print(f'Evaluator {display_name}: {elapsed_time} s')
        with open (self.save_path, 'a', encoding='utf8') as f:
            json.dump({display_name: elapsed_time}, f)
            f.write('\n')

def fn():
    pass

def clean_text(context, claim):
    word_cases = {token.lower():token for token in context.strip().split()}

    text = ' '.join(word_cases.get(token.lower(), token) for token in claim.strip().split())
    text = text.replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘', '\'').replace('`', '\'').replace('-lrb-', '(').replace('-rrb-', ')')
    text= ' '.join(each.strip()[0].capitalize()+each.strip()[1:] for each in sent_tokenize(text))
    
    return text

def score(align_fn, result_save_name):
    global datasets, datasets_validation

    results = []

    for name, dataset in datasets.items():
        sent1 = []
        sent2 = []
        true_score = []

        for example in dataset:
            sent1.append(example['document'])
            sent2.append(clean_text(example['document'], example['claim']))
            true_score.append(example['label'])

        pred_score = align_fn(sent1, sent2)[1].tolist()

        sent1_validation = []
        sent2_validation = []
        true_score_validation = []

        for example_validation in datasets_validation[name]:
            sent1_validation.append(example_validation['document'])
            sent2_validation.append(clean_text(example_validation['document'], example_validation['claim']))
            true_score_validation.append(example_validation['label'])

        pred_score_validation = align_fn(sent1_validation, sent2_validation)[1].tolist()

        thresh_result = []
        for i in range(1001):
            thresh = i / 1000
            thresh_result.append((thresh, balanced_accuracy_score(true_score_validation, [p>thresh for p in pred_score_validation])))
        
        best_thresh = sorted(thresh_result, key=lambda x: x[1], reverse=True)[0][0]

        results.append({
            'Dataset_name': name,
            'F1': [f1_score(true_score, [m>0.5 for m in pred_score])],
            'Accuracy': [accuracy_score(true_score, [m>0.5 for m in pred_score])],
            'BalancedAcc': [balanced_accuracy_score(true_score, [m>best_thresh for m in pred_score])],
            'threshold': best_thresh,
            'AUC': [roc_auc_score(true_score, pred_score)]
        })

    if not os.path.exists('./benchmarks'):
        os.makedirs('./benchmarks')

    with open(f'{result_save_name}.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def eval_align_nlg(ckpt_path, model, device, batch_size, nlg_eval_mode, comment):
    alignscore_scorer = AlignScoreScorer(ckpt_path=ckpt_path, model=model, device=device, batch_size=batch_size)
    alignscore_scorer.nlg_eval_mode = nlg_eval_mode
    
    name = f'AlignScore-{nlg_eval_mode}-{model}'

    timer = Timer()
    score(align_fn=alignscore_scorer.scorer, result_save_name='./benchmarks/' + name)
    timer.finish(name)

# def eval_bleurt(ckpt):
#     bleurt_scorer = BLEURTScorer(ckpt)
#     name = 'BLEURT'

#     timer = Timer()
#     score(align_fn=bleurt_scorer.scorer, result_save_name='./benchmarks/' + name)
#     timer.finish(name)

def eval_bertscore(model, device, batch_size):
    bertscore_scorer = BERTScoreScorer(model_type=model, metric='f1', device=device, batch_size=batch_size)
    name = f'BERTScore_{model.replace("/", "-")}_f'

    timer = Timer()
    score(align_fn=bertscore_scorer.scorer, result_save_name='./benchmarks/' + name)
    timer.finish(name)

def eval_mnli(model, device):
    mnli_scorer = MNLIScorer(model=model, device=device)
    name = f'MNLI_{model}'

    timer = Timer()
    score(align_fn=mnli_scorer.scorer, result_save_name=f'./benchmarks/' + name)
    timer.finish(name)

def eval_blanc(device, batch_size):
    blanc_scorer = BLANCScorer(device=device, batch_size=batch_size)
    name = 'BLANC'

    timer = Timer()
    score(align_fn=blanc_scorer.scorer, result_save_name='./benchmarks/' + name)
    timer.finish(name)

def run_benchmarks(args):
    if args.alignscore:
        if not args.alignscore_ckpt:
            parser.error('--alignscore-ckpt must be specified to run benchmark on AlignScore')

        eval_align_nlg( 
            ckpt_path=args.alignscore_ckpt, 
            model=args.alignscore_model, 
            device=args.device,
            batch_size=args.alignscore_batch_size,
            nlg_eval_mode=args.alignscore_eval_mode,
            comment=args.alignscore_comment
        )

    # if args.bleurt:
    #     if not args.bleurt_ckpt:
    #         parser.error('--bleurt-ckpt must be specified to run benchmark on BLEURT')
    #     eval_bleurt(ckpt=args.bleurt_ckpt)

    if args.bertscore:
        eval_bertscore(model=args.bertscore_ckpt, device=args.device, batch_size=args.bertscore_batch_size)

    if args.mnli:
        eval_mnli(model=args.mnli_ckpt, device=args.device)

    if args.blanc:
        eval_blanc(device=args.device, batch_size=args.blanc_batch_size)



if __name__ == '__main__':
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    parser = ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')

    alignscore_parser = parser.add_argument_group('AlignScore')
    alignscore_parser.add_argument('--alignscore', action='store_true', help='Run benchmark on AlignScore')
    alignscore_parser.add_argument('--alignscore-ckpt', type=str)
    alignscore_parser.add_argument('--alignscore-model', type=str, default='roberta-large', choices=['roberta-base', 'roberta-large'])
    alignscore_parser.add_argument('--alignscore-batch-size', type=int, default=32)
    alignscore_parser.add_argument('--alignscore-eval-mode', type=str, default='nli_sp', choices=['bin', 'bin_sp', 'nli', 'nli_sp', 'reg', 'reg_sp'])
    alignscore_parser.add_argument('--alignscore-comment', type=str, default='')

    # bleurt_parser = parser.add_argument_group('BLEURT')
    # bleurt_parser.add_argument('--bleurt', action='store_true', help='Run benchmark on BLEURT')
    # bleurt_parser.add_argument('--bleurt-ckpt', type=str, default='./BLEURT-20-D12')

    bertscore_parser = parser.add_argument_group('BERTScore')
    bertscore_parser.add_argument('--bertscore', action='store_true', help='Run benchmark on BERTScore')
    bertscore_parser.add_argument('--bertscore-ckpt', type=str, default='microsoft/deberta-xlarge-mnli')
    bertscore_parser.add_argument('--bertscore-batch-size', type=int, default=16)

    mnli_parser = parser.add_argument_group('MNLI')
    mnli_parser.add_argument('--mnli', action='store_true', help='Run benchmark on MNLI')
    mnli_parser.add_argument('--mnli-ckpt', type=str, default='roberta-large-mnli')

    blanc_parser = parser.add_argument_group('BLANC')
    blanc_parser.add_argument('--blanc', action='store_true', help='Run benchmark on BLANC')
    blanc_parser.add_argument('--blanc-batch-size', type=int, default=64)

    args = parser.parse_args()

    datasets = {}

    summac_benchmark = SummaCBenchmark(
        benchmark_folder = "./data/eval/summac/benchmark",
        cut='test',
        dataset_names=["cogensum", "xsumfaith", "polytope", "summeval", "frank"]
    )

    for dataset in summac_benchmark.datasets:
        name = dataset['name']
        datasets['summac_' + name] = dataset['dataset']

    datasets_validation = {}

    summac_benchmark_validation = SummaCBenchmark(
        benchmark_folder = "./data/eval/summac/benchmark",
        cut='val',
        dataset_names=["cogensum", "xsumfaith", "polytope", "summeval", "frank"]
    )

    for dataset in summac_benchmark_validation.datasets:
        name = dataset['name']
        datasets_validation['summac_' + name] = dataset['dataset']

    run_benchmarks(args)