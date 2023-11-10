import collections
from matplotlib import pyplot
import numpy as np
import torch
from climbing_ai.moonboard_dataset import MoonboardDataset, extract_batch
from matplotlib import pyplot
import matplotlib.cm as cm


def evaluate(model, call_model, dataloader, device):
    model.eval()
    grade_count = len(MoonboardDataset.grades)

    mae = 0
    mean_differences = []
    error_by_grade = np.zeros((grade_count, grade_count * 2 - 1))

    for batch in dataloader:
        compressed_batch = extract_batch(batch, device)
        logits_clsf = call_model(compressed_batch=compressed_batch, model=model)

        (
            input_ids,
            input_locations,
            masked_input_ids,
            masked_token_ids,
            masked_positions,
            attention_mask,
            grade_id,
            sequence_length,
        ) = compressed_batch

        clsf_prediction = torch.argmax(logits_clsf)

        diff = grade_id.cpu() - clsf_prediction.cpu()
        diff = diff.squeeze()
        mae += abs(diff)
        mean_differences.append(int(abs(diff)))
        error_by_grade[grade_id.cpu().squeeze(), diff + grade_count - 1] += 1

    model.train()

    errorByGradeRowSums = error_by_grade.sum(axis=1)
    errorByGradeNorm = error_by_grade / errorByGradeRowSums[:, np.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(errorByGradeNorm, cmap=cm.cool)
    ax.figure.colorbar(im)

    ax.set_xticks(np.arange(grade_count * 2 - 1))
    ax.set_xticklabels(np.arange(grade_count * 2 - 1) - (grade_count - 1))

    ax.set_yticks(np.arange(grade_count))
    ax.set_yticklabels(MoonboardDataset.grades)
    pyplot.show()

    print("MAE = {}".format(mae / len(dataloader)))

    maeDiffDictionnary = collections.Counter(mean_differences)
    maeDiffDictionnary = collections.OrderedDict(sorted(maeDiffDictionnary.items()))
    maeDiffKeyPercentage = [
        x / len(mean_differences) for x in maeDiffDictionnary.values()
    ]

    zeroGradeErrorCount = maeDiffKeyPercentage[0] * 100
    oneGradeErrorCout = maeDiffKeyPercentage[1] * 100
    correctnessCount = zeroGradeErrorCount + oneGradeErrorCout

    twoGradeErrorCount = maeDiffKeyPercentage[2] * 100
    threeGradeErrorCount = maeDiffKeyPercentage[3] * 100
    firstBoundGradeErrorCount = twoGradeErrorCount + threeGradeErrorCount

    fourGradeErrorCount = maeDiffKeyPercentage[4] * 100
    fiveGradeErrorCount = maeDiffKeyPercentage[5] * 100
    sixGradeErrorCount = maeDiffKeyPercentage[6] * 100
    secondBoundGradeErrorCount = (
        fourGradeErrorCount + fiveGradeErrorCount + sixGradeErrorCount
    )

    print(
        "Correctness Count = {{0}} : [{}%] + {{1}} : [{}%] = {}%".format(
            round(zeroGradeErrorCount, 2),
            round(oneGradeErrorCout, 2),
            round(correctnessCount, 2),
        )
    )

    print(
        "First Bound Grade Error Count = {{2}} : [{}%] + {{3}} : [{}%] = {}%".format(
            round(twoGradeErrorCount, 2),
            round(threeGradeErrorCount, 2),
            round(firstBoundGradeErrorCount, 2),
        )
    )

    print(
        "Second Bound Grade Error Count = {{4}} : [{}%] + {{5}} : [{}%] + {{6}} : [{}%] = {}%".format(
            round(fourGradeErrorCount, 2),
            round(fiveGradeErrorCount, 2),
            round(sixGradeErrorCount, 2),
            round(secondBoundGradeErrorCount, 2),
        )
    )

    pyplot.bar(maeDiffDictionnary.keys(), maeDiffKeyPercentage, 0.5, color="g")
    pyplot.plot()
