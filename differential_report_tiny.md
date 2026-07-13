# Differential datapoints: depth-gated tokens

Thresholds: learned < 0.25 nats (median over seeds), unlearned > 1.5 nats.

## Sanity gate: mean val CE by depth (must be monotone non-increasing)

| spec | seed0 | seed1 | seed2 | median |
|---|---|---|---|---|
| block3 | 1.7547 | 1.7603 | 1.7626 | 1.7393 |
| block4 | 1.7091 | 1.7180 | 1.7105 | 1.6932 |

Monotone: YES

## Depth-gated token counts

- **block3 → block4**: 5743 tokens gated (0.0382% of val stream) → `runs_lm/gated_depth8.npy`

### Examples gated at block4 (context ⟶ **token**, CE block3→block4)

- `...ge. We just wanted to taste the cake. Please don't be mad at us."

Their mom sighed. She was still angry, but she also loved Sara and` ⟶ **` Ben`** (1.75→0.14)
- `... wind. It was a noise coming from the bathroom. She got up and toddled to the door. There, she found her mom quietly humming while shower` ⟶ **`ing`** (1.76→0.10)
- `... You are very smart and organized. I admire you." Ant said.

"Thank you, Ant. You are very kind and brave. I admire` ⟶ **` you`** (1.56→0.17)
- `... was very slippery. He tried to go up the hill many times, but he could not do it. Max was sad.

Max tried one more time to go up the ` ⟶ **`ic`** (2.06→0.07)
- `... a heel lying in the grass. She hadn't seen one before so she was very curious. Abigail picked it up and was about to put it on when a v` ⟶ **`o`** (1.61→0.16)
- `...bye. And we have to listen and not touch anything. Do you understand?"

Lila says, "Yes, I understand. Let's go and knock` ⟶ **` on`** (1.89→0.20)
- `....

The little girl was so happy that she could make the bell ring. From then on, every morning she would write the word "ring" in her messy not` ⟶ **`e`** (2.03→0.19)
- `...! He quickly ate the tomato and thanked his mom for finding it. Tom enjoyed his juicy tomato and never forgot to get real tom` ⟶ **`at`** (2.01→0.09)
- `... fun to end, so she kept peddling until she was all the way around the block.

When she got back home, she felt so happy. She was so proud that she had ped` ⟶ **`al`** (2.84→0.10)
- `.... The barber smiled and talked to dad. He used the scissors and the razor carefully. He did not hurt dad. He made dad's hair shor` ⟶ **`ter`** (2.61→0.23)
- `... said. "That is not supposed to be there! We have to clean it up."

Jack and his mom found some trash bags and started to clean up the was` ⟶ **`t`** (1.91→0.23)
- `...ics and sprayed each other to cool down.Once upon a time, there was a zebra named Zara. Zara was walking in the jungle when` ⟶ **` she`** (1.52→0.16)
- `...bles with their neighbors.

One day, Tom wants to make a salad for lunch. He goes to the garden and picks some lettuce, to` ⟶ **`m`** (2.30→0.23)
- `... Your bike is so fast. I wish mine was like yours."

Timmy smiled and said, "Maybe we can race next time. Then your bike can spe` ⟶ **`ed`** (1.77→0.22)
- `... his mom's cupboard. He mixed the spices to make a special spicy powder. He sprinkled the powder onto the ro` ⟶ **`se`** (2.01→0.21)
- `...e is Sam. It's nice to meet you!" Jay smiled back and said, "My name is Jay. It's nice to meet you too!"

The giant and J` ⟶ **`ay`** (1.81→0.01)
- `...ke. They decided to do something nice for the girl.

"Here, you can have my laser. It's very cool and fun. You can pretend you're a space her` ⟶ **`o`** (1.54→0.06)
- `... Her brother said, "But I want to see fire." Lily said, "Fire can hurt us and burn things we love." Her brother said, "Okay, let's not b` ⟶ **`urn`** (1.61→0.10)
- `... sure it's nothing scary."

Mick tried to take the spear down, but it was stuck. Folly stood at a safe distance and watched anx` ⟶ **`i`** (2.11→0.08)
- `...rent colors and sizes. Tom's favorite car is a red one that can make loud noises. He likes to pretend that he is a mechanic and fi` ⟶ **`x`** (2.01→0.24)
- `... end.Once upon a time there was a little girl named Sophia who liked to swim. She had a bright yellow swimsuit and shiny yellow swim gog` ⟶ **`g`** (1.72→0.06)
- `...Once upon a time, there was a train. It was very big and long. Every day it would stretch out along the tracks. Every morning, the train cond` ⟶ **`uc`** (1.68→0.15)
- `...atching a lot. She looked at his fur and saw tiny black dots. They were fleas. Fleas are bad bugs that bite dogs and make them it` ⟶ **`ch`** (1.94→0.24)
- `... He wanted to put all his drawings in it. Tim showed the notebook to his sister, Sue.

"Look, Sue! I found a cool note` ⟶ **`b`** (2.72→0.15)
- `...ed and grumbled, but he waited patiently. He watched as Lucy quickly finished up the food and drink supply before finally saying,` ⟶ **` "`** (2.25→0.16)

