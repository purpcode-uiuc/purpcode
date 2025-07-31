# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0


def split_batch(iterable, n=1) -> list:
    ret = []
    l = len(iterable)
    for ndx in range(0, l, n):
        ret.append([iterable[i] for i in range(ndx, min(ndx + n, l))])
    return ret


from utils.litellm import *
