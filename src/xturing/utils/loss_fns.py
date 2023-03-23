from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, mask=None):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        if mask is not None:
            mask = mask.view(-1).bool()
            input = input[mask]
            target = target[mask]

        return super().forward(input, target)
