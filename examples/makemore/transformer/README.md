# A Simple GPT (Decoder-Only Transformer)

This example tries to train the same character-based GPT language model as in the
[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## How to Train?

```
go run .
```

Add `--interactive` for interactive training so that you can change learning rate
and training epochs dynamically:

```
go run . --interactive
```

## Model Architecture

The Transformer model is built in a hierarchical manner.

### Masked Self-Attention

Take embedding dimension = 2 and key/value/query vector dimension = 2 as an example:
![Masked Self-Attention Illustration](/assets/masked-self-attention-illustration.png)

### Multi-Head Attention

Take embedding dimension = 32 and head number = 2 as an example:
![Multi-Head Attention Illustration](/assets/multi-head-attention-illustration.png)

### Attention Block

Take embedding dimension = 96 and head number = 3 as an example:
![Attention Block Architecture](/assets/attention-block-architecture.png)

### Transformer

Take embedding dimension = 96 and attention block layers = 3 as an example:
![Transformer Architecture](/assets/transformer-architecture.png)

## Performance

The following shows the model performance of generating 1000 characters out of
nothing:

```
And rich, sit
Is oft, I kill bless'd.

LUCIO:
Be heaven be his for sting Burga, I am to couch'd
The blood uncieul, such in this
away our not by
Is she more fock'd comment,
For their lay; you tomb-soleman beseechstorm should night:
Spiliclant misdel'd I saids against the word, then trule, savily heads he wa love.

Second George what consinuate dead, Thallow'd,
And old unly be finds diance:
Sraw I'll not part out Romeo?
I was his haters with vale cadute, and players,
So God's king Henry whose dannatuful speak may Vatusage:
I am asnieve become's death;
On gracious jot noble your the sorthold.
Your bid brakes nothing been heaven the preperous
Infille
As behearies; follow voices as this extrems Coriolus,
Says, how he surn thee! I dangry; stcen; more what that harve, to devitor: Planting to sweet, arried;
Ha!
Good, by yonder at wherefore of his but contrable into of Sar than if list.
Come,' into faice,
No fair befuren queen cowards is a kise. foul your great for thy friar a joy.

AUFIDIUS:
A
```

This model is trained upon the [shakespeare.txt](../dataset/shakespeare.txt)
dataset with following settings:

| Context length | Attention layers | Attention heads | Embedding dimension | Batch Size | Optimizer & Learning Rate |
| --- | --- | --- | --- | --- | --- |
| 12 | 3 | 6 | 96 | 20 | Adam & 0.0003 → 0.0001 |

It consumed 5.5GB memory and reached a loss roughly at 1.60 on training set
and 1.71 on validation set after 60,000 mini-batch iterations.
