# Discriminator

Dual encoder for dialogue data.

Two lstm recurrent networks: one encodes the context (history of conversation), the other encodes the response (following utterance)

Trained to learn the probability that a given context-response pair is 'valid'.

Can be used for discriminating (`master` branch) if trained on data coming from generative dialogue systems.

Can be used for retrieval (`retriver` branch) if trained on true data only.

