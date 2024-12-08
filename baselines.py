from dataclasses import dataclass
from logging import warning
from tqdm import tqdm
from typing import Optional, Tuple

# from bleurt import score as bleurt_score
from bert_score import score as bert_score
from blanc import BlancHelp, BlancTune
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, RobertaModel, RobertaForMaskedLM
import pytorch_lightning as pl
from nltk import sent_tokenize

# class BLEURTScorer():
#     def __init__(self, checkpoint) -> None:
#         self.checkpoint = checkpoint
#         self.model = bleurt_score.BleurtScorer(self.checkpoint)

#     def scorer(self, premise:list, hypo: list):
#         assert len(premise) == len(hypo)

#         output_scores = self.model.score(references=premise, candidates=hypo, batch_size=8)
#         output_scores = [s for s in output_scores]
#         return torch.Tensor(output_scores), torch.Tensor(output_scores), torch.Tensor(output_scores)

class BERTScoreScorer():
    def __init__(self, model_type, metric, device, batch_size) -> None:
        self.model_type = model_type
        self.device = device
        self.metric = metric
        self.batch_size = batch_size

        self.model = bert_score
    
    def scorer(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        precision, recall, f1 = self.model(premise, hypo, model_type=self.model_type, lang='en', rescale_with_baseline=True, verbose=True, device=self.device, batch_size=self.batch_size)

        f1 = [f for f in f1]
        precision = [p for p in precision]
        recall = [r for r in recall]

        if self.metric == 'f1':
            return torch.Tensor(f1), torch.Tensor(f1), None
        elif self.metric == 'precision':
            return torch.Tensor(precision), torch.Tensor(precision), None
        elif self.metric == 'recall':
            return torch.Tensor(recall), torch.Tensor(recall), None
        else:
            ValueError("metric type not in f1, precision or recall.")

class MNLIScorer():
    def __init__(self, model="roberta-large-mnli", device='cuda:0', batch_size=32) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size

    def scorer(self, premise: list, hypo: list):
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score_tri = []

        for mini_batch in tqdm(batch, desc="Evaluating MNLI"):
        # for mini_batch in batch:
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(**mini_batch)
                model_output_tri = model_output.logits
                model_output_tri = self.softmax(model_output_tri).cpu()

            output_score_tri.append(model_output_tri[:,2])

        output_score_tri = torch.cat(output_score_tri)
        
        return output_score_tri, output_score_tri, output_score_tri

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            try:
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            except:
                warning('text_b too long...')
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            batch.append(mini_batch)

        return batch

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

class BLANCScorer():
    def __init__(self, device='cuda', batch_size=64) -> None:
        self.blanc_help = BlancHelp(device=device, inference_batch_size=batch_size)
        

    def scorer(self, premise, hypo):
        score = self.blanc_help.eval_pairs(premise, hypo)

        return_score = torch.tensor(score)

        return return_score, return_score, return_score
    
class AlignScoreScorer():
    def __init__(self, ckpt_path, model, device, batch_size):
        self.device = device

        self.model = AlignScoreModel.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False).to(self.device)
        self.model.eval()

        self.batch_size = batch_size

        # # self.config =
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # # self.spacy = spacy.load('en_core_web_sm')

        # # self.loss_fct
        self.softmax = nn.Softmax(dim=-1)

        # # self.smart_type = 'smart-n'
        # # self.smart_n_metric = 'f1'

        self.disable_progres_bar_in_inference = False

        self.nlg_eval_mode = None
        # # Maybe we can change this and pass as a model param?
        # # Or during the init we need it to be None?

    def scorer(self, premise, hypo):
        # assert self.nlg_eval_mode is not None, "Select NLG Eval mode!"

        if (self.nlg_eval_mode == 'bin') or (self.nlg_eval_mode == 'nli') or (self.nlg_eval_mode == 'reg'):
            return self.inference(premise, hypo)
        
        elif (self.nlg_eval_mode == 'bin_sp') or (self.nlg_eval_mode == 'nli_sp') or (self.nlg_eval_mode == 'reg_sp'):
            return self.inference_example_batch(premise, hypo)
        
    def chunks(self, lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo lists should have the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            try:
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            except:
                warning('text_b too long...')
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            batch.append(mini_batch)

        return batch
    
    def inference(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score_reg = []
        output_score_bin = []
        output_score_tri = []

        for mini_batch in tqdm(batch, desc="Evaluating", disable=self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch)
                model_output_reg = model_output.reg_label_logits.cpu()
                model_output_bin = model_output.seq_relationship_logits # Temperature Scaling / 2.5
                model_output_tri = model_output.tri_label_logits
                
                model_output_bin = self.softmax(model_output_bin).cpu()
                model_output_tri = self.softmax(model_output_tri).cpu()
            output_score_reg.append(model_output_reg[:,0])
            output_score_bin.append(model_output_bin[:,1])
            output_score_tri.append(model_output_tri[:,:])
        
        output_score_reg = torch.cat(output_score_reg)
        output_score_bin = torch.cat(output_score_bin)
        output_score_tri = torch.cat(output_score_tri)
        
        if self.nlg_eval_mode == 'nli':
            output_score_nli = output_score_tri[:,0]
            return None, output_score_nli, None
        elif self.nlg_eval_mode == 'bin':
            return None, output_score_bin, None
        elif self.nlg_eval_mode == 'reg':
            return None, output_score_reg, None
        
        return output_score_reg, output_score_bin, output_score_tri
    
    def inference_per_example(self, premise:str, hypo: str):
        """
        inference a example,
        premise: string
        hypo: string
        using self.inference to batch the process
        """
        
        premise_sents = sent_tokenize(premise)
        premise_sents = premise_sents or ['']

        n_chunk = len(premise.strip().split()) // 350 + 1
        n_chunk = max(len(premise_sents) // n_chunk, 1)
        premise_sents = [' '.join(each) for each in self.chunks(premise_sents, n_chunk)]

        hypo_sents = sent_tokenize(hypo)

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])
        
        if self.nlg_eval_mode == 'nli_sp':
            output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0] ### use NLI head OR ALIGN head
        elif self.nlg_eval_mode == 'bin_sp':
            output_score = self.inference(premise_sent_mat, hypo_sents_mat)[1] ### use NLI head OR ALIGN head
        elif self.nlg_eval_mode == 'reg_sp':
            output_score = self.inference(premise_sent_mat, hypo_sents_mat)[0] ### use NLI head OR ALIGN head

        return output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item() ### sum or mean depends on the task/aspect

    def inference_example_batch(self, premise: list, hypo: list):
        self.disable_progress_bar_in_inference = True
        assert len(premise) == len(hypo), "Premise and Hypothesis lists must have the same length!"

        out_score = []
        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc="Evaluating", disable=self.disable_progres_bar_in_inference, total=len(premise)):
            out_score.append(self.inference_per_example(one_pre, one_hypo))
        
        return None, torch.tensor(out_score), None



class AlignScoreModel(pl.LightningModule):
    def __init__(self, model, using_pretrained=True, *args, **kwargs):
        super().__init__()
        self.model = model # roberta-base OR roberta-large

        if using_pretrained:
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
        else:
            self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head

        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

        # self.need_mlm = True
        # self.is_finetune = False
        # self.mlm_loss_factor = 0.5

        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, batch):
        base_model_output = self.base_model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None
            )
        
        prediction_scores = self.mlm_head(base_model_output.last_hidden_state) ## sequence_output for mlm
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output)) ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        # total_loss = None
        # if 'mlm_label' in batch.keys(): ### 'mlm_label' and 'align_label' when training
        #     ce_loss_fct = nn.CrossEntropyLoss(reduction='sum')
        #     masked_lm_loss = ce_loss_fct(prediction_scores.view(-1, self.base_model.config.vocab_size), batch['mlm_label'].view(-1)) #/ self.con vocabulary
        #     next_sentence_loss = ce_loss_fct(seq_relationship_score.view(-1, 2), batch['align_label'].view(-1)) / math.log(2)
        #     tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch['tri_label'].view(-1)) / math.log(3)
        #     reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch['reg_label'].view(-1), reduction='sum')

        #     masked_lm_loss_num = torch.sum(batch['mlm_label'].view(-1) != -100)
        #     next_sentence_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
        #     tri_label_loss_num = torch.sum(batch['tri_label'].view(-1) != -100)
        #     reg_label_loss_num = torch.sum(batch['reg_label'].view(-1) != -100.0)

        # return ModelOutput(
        #     loss=total_loss,
        #     all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss]  if 'mlm_label' in batch.keys() else None,
        #     loss_nums=[masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num] if 'mlm_label' in batch.keys() else None,
        #     prediction_logits=prediction_scores,
        #     seq_relationship_logits=seq_relationship_score,
        #     tri_label_logits=tri_label_score,
        #     reg_label_logits=reg_label_score,
        #     hidden_states=base_model_output.hidden_states,
        #     attentions=base_model_output.attentions
        # )

        return ModelOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )
    
@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
