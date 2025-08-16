_base_ = ['../pixart_diffusion.py']
data_root = '/data/lsq/dataset/GarmentCodeData_v2'

start_config = {
    'max_pattern_len': 39,
    'max_panel_len': 37,
    'max_num_stitches': 104,
    'max_stitch_edges': 52,
    'element_size': 9,
    'edge_type_size': 7,
    'rotation_size': 4,
    'translation_size': 3,
    'explicit_stitch_tags': False,
    'img_size': 512,
    'standardize': {
        "gt_shift": {
            "outlines": [
                0.0,
                0.0,
                0.0435069315135479,
                -0.0013821386964991689,
                0.02369004860520363,
                6.655487231910229e-05,
                0.04348168149590492,
                0.00011081719276262447,
                6.126057147979736
            ],
            "rotations": [
                0.0041581252589821815,
                -0.029093468561768532,
                3.425481918384321e-05,
                0.9375355243682861
            ],
            "translations": [
                -0.010849719867110252,
                112.52392578125,
                2.6280436515808105
            ],
            "stitch_tags": [
                0.016028186306357384,
                81.98733520507812,
                0.7275835275650024
            ]
        },
        "gt_scale": {
            "outlines": [
                21.05330467224121,
                18.41973876953125,
                0.14928792417049408,
                0.04463037848472595,
                0.09293685108423233,
                0.06073087453842163,
                0.16462074220180511,
                0.060732707381248474,
                44.484989166259766
            ],
            "rotations": [
                0.051953211426734924,
                0.18981429934501648,
                0.20662523806095123,
                0.19682729244232178
            ],
            "translations": [
                27.13235855102539,
                33.73225784301758,
                23.4309139251709
            ],
            "stitch_tags": [
                22.296857833862305,
                45.6525993347168,
                17.834659576416016
            ]
        }
    },
    'augment': False,
    'panel_classification': './configs/data_configs/panel_order.json',
    'body_root': '/data/lsq/dataset/5000_body_shapes_and_measures/meta'
}


data = dict(
    type='GarmentDetrDataset',
    root_dir=data_root,
    data_json='./data_info/dataset_v2_info.json',
    known_split='./data_info/misc/GarmentCodeData_v2_official_train_valid_test_data_split.json',
    start_config=start_config,
    load_text_feat=False,
    condition='text',
    load_pattern_vector=False,
    load_tokenizer_feat=False,
    load_body_params=True,
    load_sketch=True,
    body_caching=True,
    gt_caching=True,
    feature_caching=False,
)

datawrapper = 'SewingLDMDatasetWrapper'

loss = dict(
    loss_components=['shape', 'loop', 'rotation', 'translation', 'stitch', 'free_class', 'edge_type', 'edge_mask'],  # stitch, stitch_supervised, free_class, edge_type
    quality_components=['shape', 'discrete', 'rotation', 'translation', 'stitch', 'free_class'],  # stitch, free_class
    loss_weight_dict={
        'shape_loss_weight': 1.,
        'loop_loss_weight': 1.,
        'edge_loss_weight': 1.,
        'rotation_loss_weight': 1.,
        'translation_loss_weight': 1.
    },
    stitches='ce',  # ce, simple
    lepoch=0,
    eos_coef=0.1,
    aux_loss=False,
    panel_origin_invariant_loss=False,
    panel_order_inariant_loss=False,  # False to use ordering as in the data_config
    epoch_with_order_matching=0,
    epoch_with_stitches=75, 
    order_by='shape_translation'  # placement, translation, stitches, shape_translation
)

image_size = 512

tokenizer_token_size = start_config['max_pattern_len'] * start_config['max_panel_len']
continous_channels = (start_config['element_size'] + 3) + 7
discrete_channels = (start_config['edge_type_size'] + 2 + 1)
use_tokenizer = True

# model setting
tokenizer = 'Dress_FSQTOKENIZER'
tokenizer_config = dict(
    bound=True,
    encoder=dict(
        drop_rate=0.2,
        num_blocks=15,
        hidden_dim=256,
        token_inter_dim=1024,
        hidden_inter_dim=1024,
        dropout=0.0,
    ),
    decoder=dict(
        num_blocks=15,
        hidden_dim=256,
        token_inter_dim=1024,
        hidden_inter_dim=1024,
        dropout=0.0,
    ),
    codebook=dict(
        token_num=256,
        token_dim=6,
        token_class_num=3,
        ema_decay=0.9,
    ),
    loss_keypoint=dict(
        cloth_loss_w=1.0, 
        e_loss_w=5.0,
        beta=0.05,)
)
load_tokenizer_from='./models/auto_encoder.pth'
scaling_factor=1.8407

# model setting
model_token_size = tokenizer_config['codebook']['token_num']
in_channels = tokenizer_config['codebook']['token_dim']
model = 'PixArt_XL_2'     # model for multi-scale training
fp32_attention = False
load_from = None
load_sewingldm_from = './models/sewingldm.pth'
aspect_ratio_type = 'ASPECT_RATIO_512'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = False     # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers=4
train_batch_size = 256
validation_size = 2
num_epochs = 50
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=20)
save_model_epochs=10
save_model_steps=10000

log_interval = 20
eval_sampling_steps = 2500
work_dir = 'output/debug'

# multimodalnet setting
dit_trainable_param_pattern = r".*attn.*proj.*"
multimodalnet_path = None
control_scale = 1.0
