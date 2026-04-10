from datasets.ravdess import RAVDESS

# def get_training_set(opt, spatial_transform=None, audio_transform=None, fisher_indices_path=None):
#     assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))

#     if opt.dataset == 'RAVDESS':
#         training_data = RAVDESS(
#             opt.annotation_path,
#             'training',
#             spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform, fisher_indices_path=fisher_indices_path)
#     return training_data


# def get_validation_set(opt, spatial_transform=None, audio_transform=None, fisher_indices_path=None):
#     assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))

#     if opt.dataset == 'RAVDESS':
#         validation_data = RAVDESS(
#             opt.annotation_path,
#             'validation',
#             spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform, fisher_indices_path=fisher_indices_path)
#     return validation_data


# def get_test_set(opt, spatial_transform=None, audio_transform=None, fisher_indices_path=None):
#     assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
#     assert opt.test_subset in ['val', 'test']

#     if opt.test_subset == 'val':
#         subset = 'validation'
#     elif opt.test_subset == 'test':
#         subset = 'testing'
#     if opt.dataset == 'RAVDESS':
#         test_data = RAVDESS(
#             opt.annotation_path,
#             subset,
#             spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform , fisher_indices_path=fisher_indices_path)
#     return test_data

# dataset.py — make sure data_type is forwarded here
def get_training_set(opt, spatial_transform=None, fisher_indices_path=None, audio_transform=None):
    return RAVDESS(
        annotation_path=opt.annotation_path,
        subset='training',
        spatial_transform=spatial_transform,
        audio_transform=audio_transform,
        data_type=opt.data_type,          # ← this is the likely missing line
        fisher_indices_path=fisher_indices_path,
        audio_channels=opt.audio_channels,  # ← and this if using proportional channel selection
    )

def get_validation_set(opt, spatial_transform=None, fisher_indices_path=None, audio_transform=None):
    return RAVDESS(
        annotation_path=opt.annotation_path,
        subset='validation',
        spatial_transform=spatial_transform,
        audio_transform=audio_transform,
        data_type=opt.data_type,          # ← same here
        fisher_indices_path=fisher_indices_path,
        audio_channels=opt.audio_channels,  # ← and here if using proportional channel selection
    )

def get_test_set(opt, spatial_transform=None, fisher_indices_path=None, audio_transform=None):
    return RAVDESS(
        annotation_path=opt.annotation_path,
        subset='testing',
        spatial_transform=spatial_transform,
        data_type=opt.data_type,          # ← and here
        fisher_indices_path=fisher_indices_path,
        audio_channels=opt.audio_channels,  # ← and here if using proportional channel selection
    )