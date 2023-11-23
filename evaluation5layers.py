from transformers import AutoModelForCausalLM, AutoTokenizer
from dataloader import *
from utils import *
from model_vit_bert import ViTConfigCustom, ViTModelCustom, CustomVEDConfig, CustomVisionEncoderDecoder
from training_script_vit_bert_5layers import LightningModel
from tqdm import tqdm
import os

# models: Encoder    
encoder = ViTModelCustom(config=ViTConfigCustom(hidden_size=576), pretrain_4k='vit4k_xs_dino', freeze_4k=True)

# decoder
decoder_model_name="emilyalsentzer/Bio_ClinicalBERT"
decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, is_decoder=True, add_cross_attention=True)
tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

# encoder decoder model
model=CustomVisionEncoderDecoder(config=CustomVEDConfig(),encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

lightning_model = LightningModel(model, tokenizer, model_lr=1e-2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ckpt="/scratch/ss4yd/logs_only_vit_bert_fe/my_model/version_7/checkpoints/epoch=6-val_loss=0.82-step=4340.00.ckpt"
lightning_model.load_state_dict(torch.load(ckpt,map_location=device)['state_dict'])
lightning_model=lightning_model.to(device)
lightning_model.eval()

## Load Data
import pandas as pd
df_path='/home/ss4yd/nlp/final_more_female_data.pickle'
df=pd.read_pickle(df_path)

df=df[df.dtype=='test']

# function
def generate_func(lm, reps_path, max_length=128, num_beams=2, do_sample=True):
    pixel_values=torch.load(reps_path).unsqueeze(0)
    pixel_values=pixel_values.to(device)
    gencap=lm.model.generate(pixel_values, max_length=max_length, num_beams=num_beams, do_sample=do_sample)

    decoded_cap=tokenizer.decode(gencap[0])
    remove_sptokens=decoded_cap[6:decoded_cap.find('[SEP]')]

    # print(f'Generated Note: \n {remove_sptokens}')

    return remove_sptokens

print(lightning_model.device)
tqdm.pandas()
df['pred_notes'] = df['reps_path'].progress_apply(lambda x: generate_func(lightning_model, x))

df.to_pickle(os.path.join('/'.join(ckpt.split('/')[:-1]),'saved_df.pickle'))