# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Losses contains common loss computation."""
#from .weighted_binary_crossentropy_loss import weighted_binary_crossentropy_loss
from .weighted_sparse_categorical_crossentropy_loss import weighted_sparse_categorical_crossentropy_loss
#from .weighted_sparse_softmax_cross_entropy_with_logits import weighted_sparse_softmax_cross_entropy_with_logits