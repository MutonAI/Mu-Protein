
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion('prot_mlm')
class ProteinMaskedLMCriterion(FairseqCriterion):
    def __init__(self, task, tpu=False):
        super().__init__(task)
