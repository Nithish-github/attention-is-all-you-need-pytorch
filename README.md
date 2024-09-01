# Attention is all you need: A Pytorch Implementation Modifided to Generate Tan waveform from sine and cos.

## 1. "sine_cos_tan.ipynb" has juputer notebook implementation <br> 2. To train the model run "python3 train_wave.py"

## Key changes 

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2):
            super(TransformerModel, self).__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
            self.output_projection = nn.Linear(d_model, output_dim)
        
        def forward(self, src):
            src = self.input_projection(src)
            transformer_output = self.transformer(src.unsqueeze(0), src.unsqueeze(0))
            output = self.output_projection(transformer_output.squeeze(0))
            return output

Created a TransfomerModel where it can get 2 seq(sine,cos) input and provide 1seq(tan) output.