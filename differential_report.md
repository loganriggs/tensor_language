# Differential datapoints: depth-gated tokens

Thresholds: learned < 0.5 nats (median over seeds), unlearned > 1.5 nats.

## Sanity gate: mean val CE by depth (must be monotone non-increasing)

| spec | seed0 | seed1 | seed2 | median |
|---|---|---|---|---|
| attn1 | 2.4509 | 2.4461 | 2.4490 | 2.4284 |
| attn2 | 2.3013 | 2.2900 | 2.2749 | 2.2613 |
| attn3 | 2.2332 | 2.2393 | 2.2362 |

Monotone: YES

## Depth-gated token counts

- **attn1 → attn2**: 136740 tokens gated (0.9102% of val stream) → `runs_lm/gated_depth2.npy`

### Examples gated at attn2 (context ⟶ **token**, CE attn1→attn2)

- `... amazed. It was so beautiful and she wanted to explore more. She knew that packing those scissors was a good idea and it paid off in the end. J` ⟶ **`an`** (3.81→0.04)
- `... day, Dave wanted to go outside and play. He grabbed his ball and rolled it across the grass. He rolled it back and forth and laughed with glee` ⟶ **`.`** (1.59→0.28)
- `.... He saw the celery. He bit it. It was green and crunchy. He liked it too. He ate his soup. He ate his celer` ⟶ **`y`** (1.80→0.42)
- `...

The little girl never got the pink dress she wanted. She was very sad. The mean man had stolen her money, so she never got to care for her pocket m` ⟶ **`one`** (2.61→0.50)
- `... brown gate. Every day, Suzy went outside and opened the gate. She would go to the market and sell her vegetables.

One day, S` ⟶ **`u`** (1.98→0.04)
- `... there was a little ant who loved to crawl around. One day, the ant saw a large cherry on the ground. It looked so delicious, but the a` ⟶ **`nt`** (1.55→0.22)
- `.... She had so much fun!

When she was done playing, she put her special suit back on and walked away with a smile. She vowed to always remember her special su` ⟶ **`it`** (1.60→0.12)
- `... the new staff and all of her people in the castle were glad too. They all celebrated by singing and dancing together.

The prin` ⟶ **`c`** (1.52→0.27)
- `... Girl feel guilty. She said, â€œMama, Iâ€™m sorry. I shouldnâ€™t have asked for another joke. I should have let the but` ⟶ **`ter`** (1.81→0.38)
- `... waited patiently for the sun to set. As soon as the sky began to turn orange, he blew into the horn and made a beautiful, tinkl` ⟶ **`ing`** (2.23→0.32)
- `... for them. She hugged them and said, "Hello, my sweeties! I'm so glad you're here. Come inside, I have a surprise for you."

J` ⟶ **`en`** (2.95→0.38)
- `... with others. It will make them happy and you will feel happy too."

Lily thought about what her mom said and decided to share the blueberries with Max. They both ate the blue` ⟶ **`ber`** (2.40→0.16)
- `...es and cups in the dishwasher. Mom said the dishwasher made them clean with hot water and soap.

One day, Anna and Ben found a big cho` ⟶ **`c`** (1.57→0.21)
- `...uly care and raise others with love, you can witness the beauty of the world grow right before your eyes.

So, the little girl conclud` ⟶ **`ed`** (2.53→0.20)
- `...ng songs.

After a while, the girl had to go home. But she thanked the ghost for the fun day. "You are so jolly," she said. The g` ⟶ **`h`** (3.32→0.00)
- `.... They were happy and surprised. They petted the turkey and said, "Hello, turkey. You are a friend. We understand you now." The t` ⟶ **`ur`** (3.89→0.02)
- `... with us?"

The boy stopped his bike and looked at them. He shook his head.

"No, thank you," he said. "I am a real p` ⟶ **`ir`** (3.83→0.16)
- `... thanked her mom and ran back to the garden.

She put the vase on the table and called the bird. The bird saw the roses and perked up. It flew to the v` ⟶ **`ase`** (2.20→0.01)
- `...ffee. She gave one to the boy and said, "Let's sit together and talk."

The boy smiled and took the coffee. Then they both sat down and t` ⟶ **`al`** (1.60→0.36)
- `...ning the radish changed back to red and the house stayed clean.

The radish was happy. It was always able to be part of a tidy house. The r` ⟶ **`ad`** (3.26→0.00)
- `..., his whispers flew away, creating a little whispering wind. He loved the feeling of his whispers floating away, so he decided to keep the wh` ⟶ **`is`** (2.81→0.09)
- `... hearts.Anna and Ben were twins who liked to play with their toys. One day, they found a big box in the garage. They opened it and saw many old things that be` ⟶ **`l`** (2.48→0.46)
- `... had lost his glove. Tom was feeling sad and Fred was determined to help him. He asked all his other friends if anyone had a spare glove to l` ⟶ **`end`** (3.30→0.47)
- `...y for him to run on the icy ground. Sam was having fun too, but he was not as fast as Tim.

Suddenly, the ball flew high in the air and land` ⟶ **`ed`** (1.85→0.25)
- `... her head and the big man scooped her up. He held her close and carried her over to a bench where he sat her down.

He watched as B` ⟶ **`et`** (3.29→0.14)

- **attn2 → attn3**: 29117 tokens gated (0.1938% of val stream) → `runs_lm/gated_depth3.npy`

### Examples gated at attn3 (context ⟶ **token**, CE attn2→attn3)

- `...ll and hit her head on the carpet. At first, it seemed harmless, but soon she started to feel sick. The family took her to the ho` ⟶ **`s`** (1.76→0.36)
- `... slide in her backyard. Every time she got the chance, she would slide down the slippery slide. One day, Jen was feeling hungry and wanted something to` ⟶ **` eat`** (1.59→0.42)
- `... hit the ice hard, but it did not crack. He hit it again, and again, but nothing happened.

"Be careful, Ben!" Lily said. "You m` ⟶ **`ight`** (1.81→0.29)
- `..., there was a big pond. Emma liked to look at the way the water was so smooth and shiny.

Sometimes, Emma would pick up a few peb` ⟶ **`ble`** (1.55→0.40)
- `...an bullies. They wanted to take Timmy's toy truck away. Timmy was scared and didn't know what to do. But then, he remembered the pigeon he saw earl` ⟶ **`i`** (2.17→0.44)
- `... time, there was a small pebble named Pebby. Pebby lived in a big garden with many other pebbles. One day, Pebby saw a big rock tal` ⟶ **`king`** (1.68→0.34)
- `... at the little boy and smiled. He thought that was a very kind thing to say. The driver of the bus thanked the little boy again, and the bus drove off into the suns` ⟶ **`et`** (1.66→0.48)
- `...ook their heads and said, "it's too bad that she was so worried." 

The end.Grandad loved fixing cars. He spent hours and h` ⟶ **`our`** (2.02→0.23)
- `... Mr. Bear?" Anna asked, pouring water into a cup.

"Yes, please, Anna. You are very kind," Tom said, making his bear tal` ⟶ **`k`** (1.69→0.43)
- `.... Emma didn't like learning, she thought it was too hard and she wanted to play instead.

One day, Emma's mum said to her that she should stu` ⟶ **`dy`** (1.93→0.39)
- `...The sun comes out. It makes the park bright. It makes them warm. They are happy.

The moral of the story is: Sharing and help` ⟶ **`ing`** (1.75→0.48)
- `... Timmy's mom made him a special oatmeal with rare blueberries in it. Timmy thought it was the best oatmeal he ever had.

After bre` ⟶ **`ak`** (2.67→0.49)
- `... reach the black car. They peek through the window. They see a man in the driver's seat. He is wearing a black hat and a black co` ⟶ **`at`** (1.86→0.31)
- `... soft toys and dolls. One day, she was playing with her favorite teddy bear when her mom came in and said, "Lily, we need to decorate the Christ` ⟶ **`m`** (2.57→0.20)
- `... friendly insects. She loved exploring and discovering different things.

Suddenly, Daisy heard a noise. She saw two caterp` ⟶ **`ill`** (1.57→0.15)
- `...! Look at the helicopter!"

One day, Timmy and his mom went to the park. As they were walking, they heard a loud noise. It was a great big hel` ⟶ **`ic`** (1.80→0.50)
- `... saw a thin snake on the grass. Tom wanted to touch it, but Sam was afraid.

"Look, a snake!" Tom said. "Let's prep` ⟶ **`are`** (2.28→0.44)
- `.... He sees the aeroplane and wants it for himself. He climbs the tree and takes the aeroplane. He does not give it back to Anna and` ⟶ **` Ben`** (2.23→0.35)
- `...hers and a sweet face. Lily wanted to be friends with the pigeon.

So, she climbed up on the bench next to the pigeon. The pig` ⟶ **`e`** (1.65→0.39)
- `... the tea and started to feel better. Nutty was happy that he could help his friend.Once upon a time, there was a little girl named Lily. One day, she saw a shiny oy` ⟶ **`st`** (1.87→0.42)
- `... increasing in size.

The people were scared and didn't know what to do. But then one wise, old woman told them that the sea could only get big` ⟶ **`ger`** (2.32→0.22)
- `...Sara loved her pony, Spot. He was white with black spots, and he was very fast and strong. Sara liked to ride him in the field and brus` ⟶ **`h`** (2.10→0.39)
- `... explore and find new things in his garden.

This time, he discovered something green and round. He picked it up and took it to his mom. It was an ol` ⟶ **`ive`** (1.57→0.46)
- `.... "These toys are not for sale. They are part of my collection. I only show them to people who love toys like you do."

Sara and Tom understand` ⟶ **`.`** (1.85→0.29)
- `...y.

When the tea was gone, John and Mandy said goodbye and went on their way. They always felt very special when they spent time together.Once upon a time` ⟶ **` there`** (1.86→0.46)

