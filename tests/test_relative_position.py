import torch
from climbing_ai.wolfbert import Relative2dPositionBias


def test_relative_position():
    positions = torch.tensor(
        [
            [-100, -100],
            [15, 85],
            [25, 50],
            [30, 15],
            [30, 25],
            [35, 65],
            [50, 40],
            [-100, -100],
            [-100, -100],
            [-100, -100],
        ]
    )

    xy_maximum = torch.max(positions, dim=0)[0]
    distance_max = torch.sqrt(torch.sum(torch.pow(xy_maximum, 2)))
    max_len = 10

    query = torch.rand(size=(max_len, max_len))
    key = torch.rand(size=(max_len, max_len))

    weight = torch.einsum("i d, j d -> i j", query, key)

    relative_bias = Relative2dPositionBias(
        1,
        num_buckets=32,
        x_max_distance=xy_maximum[0],
        y_max_distance=xy_maximum[1],
    )
    out = relative_bias.forward(weight[None, :], positions[None, :])

    assert out.shape == torch.Size((1, max_len, max_len))
