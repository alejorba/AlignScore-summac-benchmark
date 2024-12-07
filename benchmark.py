# TODO
# - Add default ckpt path for alignscore
# - Implement align_func for all metrics

from argparse import ArgumentParser
import json
import time

from summac.benchmark import SummaCBenchmark

class Timer():
    def __init__(self):
        self.start = time.time()
        self.save_path = 'exp_results/time.json'

    def finish(self, display_name):
        elapsed_time = time.time() - self.start
        print(f'Evaluator {display_name}: {elapsed_time} s')
        with open (self.save_path, 'a', encoding='utf8') as f:
            json.dump({display_name: elapsed_time}, f)
            f.write('\n')

def fn():
    pass

def score(align_fn):
    global datasets

    pass

def eval_align_nlg(nlg_eval_mode, ckpt_path, base_model, device, comment):
    # get score fn
    timer = Timer()
    score(align_func=fn)
    timer.finish(f'AlignScore-{nlg_eval_mode}-{base_model}')

def eval_bleurt(ckpt):
    # get score fn
    timer = Timer()
    score(align_func=fn)
    timer.finish('BLEURT')

def eval_bertscore(model, device, batch_size):
    # get score fn
    timer = Timer()
    score(align_func=fn)
    timer.finish(f'BERTScore_{model.replace('/', '-')}_f')

def eval_mnli(model, device):
    # get score fn
    timer = Timer()
    score(align_func=fn)
    timer.finish(f'MNLI_{model}')

def eval_blanc(device, batch_size):
    # get score fn
    timer = Timer()
    score(align_func=fn)
    timer.finish('BLANC')

def run_benchmarks(args):
    if args.alignscore:
        if not args.alignscore_ckpt:
            parser.error('--alignscore-ckpt must be specified to run benchmark on AlignScore')

        eval_align_nlg(
            nlg_eval_mode=args.alignscore_eval_mode, 
            ckpt_path=args.alignscore_ckpt, 
            base_model=args.alignscore_model, 
            device=args.device, tasks=args.tasks,
            comment=args.alignscore_comment
        )

    if args.bleurt:
        if not args.bleurt_ckpt:
            parser.error('--bleurt-ckpt must be specified to run benchmark on BLEURT')
        eval_bleurt(ckpt=args.bleurt_ckpt)

    if args.bertscore:
        eval_bertscore(model=args.bertscore_ckpt, device=args.device, batch_size=args.bertscore_batch_size)

    if args.mnli:
        eval_mnli(model=args.mnli_ckpt, device=args.device)

    if args.blanc:
        eval_blanc(device=args.device, batch_size=args.blanc_batch_size)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')

    alignscore_parser = parser.add_argument_group('AlignScore')
    alignscore_parser.add_argument('--alignscore', action='store_true', help='Run benchmark on AlignScore')
    alignscore_parser.add_argument('--alignscore-model', type=str, default='roberta-large', choices=['roberta-base', 'roberta-large'])
    alignscore_parser.add_argument('--alignscore-ckpt', type=str)
    alignscore_parser.add_argument('--alignscore-eval-mode', type=str, default='nli_sp', choices=['bin', 'bin_sp', 'nli', 'nli_sp', 'reg', 'reg_sp', 'smart-n', 'smart-l'])
    alignscore_parser.add_argument('--alignscore-comment', type=str, default='')

    bleurt_parser = parser.add_argument_group('BLEURT')
    bleurt_parser.add_argument('--bleurt', action='store_true', help='Run benchmark on BLEURT')
    bleurt_parser.add_argument('--bleurt-ckpt', type=str)

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

    for dataset in summac_benchmark:
        name = dataset['name']
        datasets['summac_' + name] = dataset['dataset']

    run_benchmarks(args)