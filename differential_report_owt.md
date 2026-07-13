# Differential datapoints: depth-gated tokens

Thresholds: learned < 0.5 nats (median over seeds), unlearned > 1.5 nats.

## Sanity gate: mean val CE by depth (must be monotone non-increasing)

| spec | seed0 | seed1 | seed2 | median |
|---|---|---|---|---|
| attn1-mix10 | 4.6803 | 4.6801 | 4.6713 | 4.6672 |
| attn2-mix10 | 4.6638 | 4.6911 | 4.6944 | 4.6711 |
| **WARNING** | attn2-mix10 mean CE above attn1-mix10 — optimization failure? | | |
| attn3-mix10 | 4.6156 | 4.6236 | 4.6531 | 4.6148 |

Monotone: **NO — fix before believing gates**

## Depth-gated token counts

- **attn1-mix10 → attn2-mix10**: 4845 tokens gated (0.0227% of val stream) → `runs_lm/gated_depth2.npy`

### Examples gated at attn2-mix10 (context ⟶ **token**, CE attn1-mix10→attn2-mix10)

- `... threatening early on.

Positive early signs were soon forgotten after the interval, though, with Nenad Kristicic sent off for a second bookable offence just mom` ⟶ **`ents`** (1.74→0.07)
- `... such as those of Alfa Romeo, Ferrari, or Renault. After having virtually disappeared by the early 1980s, factory teams made a comeback in the 1990` ⟶ **`s`** (1.51→0.48)
- `...ment by 1.5 to 2 percent. These findings were notable because they were comparable to earlier estimates from the time series literature, which relied on variation over time rather than across states to est` ⟶ **`imate`** (1.51→0.46)
- `... in worsening standards of care for the brave men and women that served our country – and all at a higher cost to the American taxpayer.”

The American Federation of Government Employ` ⟶ **`ees`** (2.02→0.35)
- `... most surplus of staff will be put on leave, as anyone with an ounce of clout will have found a way to be put on their department’s "essential" list. The few who couldn` ⟶ **`’`** (1.97→0.23)
- `... an eminent scientist, in relation to the "moons" circling around Mars.After carefully weighing up the evidence he concludes that they are both hollow and therefore artific` ⟶ **`ial`** (2.47→0.33)
- `... gaudy splendor.

While Apple and its app-developers may need to find ways to not annoy Apple Watch wearers, the wearers themselves may be a source of ann` ⟶ **`oy`** (1.55→0.44)
- `....S. Secret Service Special Agent in Charge Eric P. Zahren of the Pittsburgh Field Office today announced the filing of a criminal complaint in Pittsb` ⟶ **`ur`** (1.80→0.35)
- `... that Verizon built just for him, he tried teaching himself to cook from cookbooks and online recipes. It didn’t work.

“I struggled with getting the whole recipe download` ⟶ **`ed`** (1.85→0.38)
- `..."In Bavaria, this is not totally uncommon. White sausage and a beer for breakfast; this is something that is still practiced in this country quite a lot," he expl` ⟶ **`ains`** (1.57→0.38)
- `... that purport to be her have been confirmed, according to Varon.)

Could such a woman, when she was about 20 years old, have found a place in the Davis household in Richm` ⟶ **`ond`** (3.45→0.11)
- `.... That is, the Senators and Representatives will be busy raising money from commercial interests so they can keep their jobs. There won’t be much time to change anything about misallocated public bud` ⟶ **`g`** (1.68→0.38)
- `... sending in performance teams to reform programs; replacing existing management; demanding improvement action plans; and cutting program budgets or eliminating programs entirely. Obama will also experiment with giving government manag` ⟶ **`ers`** (2.37→0.37)
- `... physically. So how would travelers from Earth survive a five-year mission through space without the benefit of the artificial gravity present in the Enterprise?

One solution is artific` ⟶ **`ial`** (2.59→0.49)
- `... for our newsletter Get our newsletter, Dear Penn, delivered to your inbox every weekday morning. Sign Up

For other opponents, their criticism of sanct` ⟶ **`uary`** (2.77→0.29)
- `... high performance 3D graphics for the modern game player. Sammy has announced that the new machine is expected to launch in pachislot parlors across Japan beginning in fall, 2016.

The strong vis` ⟶ **`ual`** (1.55→0.47)
- `...″]

Advertisement

But as rumors swirled that the Russians had hacked the U.S. electric grid successfully, the company issued another statement Saturday with further clarific` ⟶ **`ations`** (1.79→0.42)
- `... leakage rates exceeding 3.2 percent offset natural gas’ advantage in the short term, Sweeney said.

Assertions that natural gas from fracking has a lower greenhouse` ⟶ **` gas`** (1.95→0.35)
- `... some seating for fans in the 2018 season, Whitman said, with it reopening for 2019. The east side is projected to be finished for the 2020 season.

While he decl` ⟶ **`ined`** (1.52→0.41)
- `...­tices, learn from mis­takes, and bet­ter pro­tect nation­al secur­ity, as well as avoid­ing the scan­dal and embar­rass` ⟶ **`­`** (2.66→0.49)
- `... developed above. As final step of phase 1, the bots unfollowed users which were obviously spam/bot accounts in order to decrease their following/followed ratio. To investig` ⟶ **`ate`** (1.54→0.16)
- `...ling of the economy, health care and foreign policy were particular areas of concern for most respondents. On health care, Obama is seen as having strengthened the for-profit insurance industry with little benefit for ord` ⟶ **`inary`** (1.53→0.37)
- `...20:19 that moment.

20:21

20:25 And then one last one.

20:26 Avoid the unexpected, especially the unexpected and

` ⟶ **`20`** (5.55→0.50)
- `... where it’s not just about going through the paces but actually figuring the problem out.

You’re using a new engine, and it seems to me it’d take a lot of effort to bas` ⟶ **`ically`** (1.54→0.40)
- `... latest news, every morning — in your inbox. Email Sign Up By signing up you agree to receive email newsletters or alerts from POLITICO. You can unsubsc` ⟶ **`ribe`** (1.83→0.06)

- **attn2-mix10 → attn3-mix10**: 10286 tokens gated (0.0482% of val stream) → `runs_lm/gated_depth3.npy`

### Examples gated at attn3-mix10 (context ⟶ **token**, CE attn2-mix10→attn3-mix10)

- `...ock.

AMY GOODMAN: That was Elliott Abrams—he served as assistant secretary of state for human rights and humanitarian affairs under President Reag` ⟶ **`an`** (1.89→0.30)
- `... brilliance then disappeared. But they are also, as fascists and white nationalists clearly understand, powerful laboratories for inseminating and spreading darker ideas.

It’s hard` ⟶ **` to`** (1.51→0.48)
- `...5–81.

Neumark, D., and W. Wascher. 1996. “The Effects of Minimum Wages on Teenage Employment and Enroll` ⟶ **`ment`** (1.67→0.22)
- `... sex” and “the other sex”), and the only reasonable reading of the language used throughout the relevant regulatory section is that it references male and female. Read plainly then, [the law` ⟶ **`]`** (1.70→0.46)
- `... idea where this f--king place was," said 19-year-old Alex McCormick to News.com.au on Tuesday, referring to Syria's second-largest port city on the Mediter` ⟶ **`r`** (1.80→0.44)
- `...ated 400,000 workers will lose their unemployment benefits during the first two weeks of March if the extension is not quickly approved. If congressional inaction continues, that number will grow expon` ⟶ **`entially`** (2.83→0.15)
- `...
Although Russian firms saw soaring revenues, Western sanctions did have an impact on the industry, though their effects are not readily apparent from the revenue data.

Sanctions comb` ⟶ **`ined`** (1.50→0.43)
- `...olen said, air pollution exacerbates the effects of asthma and can be the cause of an asthma attack. In Los Angeles, one of the cities with the worst air poll` ⟶ **`ution`** (1.51→0.36)
- `...22.com website includes the following bullet points to speak against IM 22:

Politically-connected special interest groups are up to no good. Measure 22 will allow big spending politic` ⟶ **`ians`** (1.62→0.46)
- `...icles to their own benefit. Image courtesy Jake Appelbaum/ Virgil Griffith

If you've ever used the online encyclopedia Wikiped` ⟶ **`ia`** (1.76→0.32)
- `...ian Jerry Apps was growing up on his parent’s farm in Wild Rose.

"I recall from my childhood the dust storms that visited this part of Wiscons` ⟶ **`in`** (1.58→0.23)
- `... think it's something from out of the past that doesn't exist in the present anymore. It's one of those old things that was widely accepted -- and then a lot of smart people said, 'Why?` ⟶ **`'`** (1.67→0.47)
- `... government," Huckabee told CBS News.

Marriage is the foundational form of government? What the hell does that even mean? Was the American revolution fought over high divor` ⟶ **`ce`** (1.61→0.38)
- `...
The Tribune and the Detroit News joined the Union Leader of Manchester, N.H., the Winston-Salem (N.C.) Journal and the Richm` ⟶ **`ond`** (1.71→0.08)
- `...s, Shabbat mode will often feature the ability to adjust the temperature of the oven without any feedback to the operator of the oven.[11] According to the prevailing Orth` ⟶ **`od`** (2.36→0.36)
- `...In an unrelated case, the High Court ruled in April that the Hong Kong government should grant civil servant welfare benefits to the spouses of civil servants married to same-se` ⟶ **`x`** (1.69→0.36)
- `... Skills tab now have tooltips that show level and attribute requirements, plus some other details

* The sidebar now shows your attribute requirements if they aren't met

` ⟶ **`*`** (4.13→0.13)
- `... driver when the vehicle is inadvertently crossing highway lane markers (with some cars further having the ability to either “nudge” the car back into place via selective braking). Mean` ⟶ **`while`** (1.55→0.26)
- `... Recession ended.

Out-migration from Illinois has spiked as taxes have shot up and job creation has faltered. During the first three years of Illino` ⟶ **`is`** (1.51→0.06)
- `... -0.420 Raiders 31 31 29 26 25 25 26 27 24 28 2.358 -0.171 Redskins 28 27 30 29 24 23 30 26 28 30 2.377 0` ⟶ **`.`** (2.07→0.50)
- `....7 percent of people who voted were Democrats, 31.9 percent were African-Americans … and they did make up some ground. There were 1,136 Democrats over their 2012 number.”

Dem` ⟶ **`ocrat`** (1.63→0.25)
- `...ian?"

The site then asked if he understood Russian.

"R u kidding?" wrote Guccifer 2.0.

In the same interview, when forced to ans` ⟶ **`we`** (2.13→0.46)
- `... the hardware of an actual machine. A supercomputer can solve problems more quickly than a Turing machine when they can be solved, but it cannot solve any problem that a Turing machine cannot sol` ⟶ **`ve`** (1.85→0.40)
- `... a mention during a speech by the primate of the OCA , Metropolitan Jonah Paffhausen, in April 2009.[24]

On September 8, 2018, the Orth` ⟶ **`od`** (2.12→0.39)
- `... IGR, we polled the writers, the gamers, the communities, we argued and opined about how to create a list of ten titles from the mass of possibilities. We wond` ⟶ **`ered`** (2.49→0.49)

