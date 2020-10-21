from tqdm import tqdm
import torch
from torch import optim
import os
import wandb
import ipdb
import numpy as np
from utils.utils import load_unk_model, create_loaders, kwargs_perturb_loss, save_image_to_wandb
import glob
from cnn_models.mnist_ensemble_adv_train_models import load_model

def eval(args, attacker, test_loader=None, list_classifiers=[], logger=None, epoch=0):
    if test_loader is None:
        _, test_loader = create_loaders(args)

    all_acc = []
    for name, classifier in list_classifiers.items():
        acc = eval_classifier(args, attacker, test_loader, classifier, name)
        all_acc.append(acc)
        if logger is not None:
            logger.write({"%s/acc"%name: acc}, epoch)

    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    if logger is not None:
        logger.write({"mean_acc": mean_acc, "loss": acc}, epoch)

    return mean_acc, std_acc, all_acc



def adv_data(args, attacker, data_loader):
    # Create adversarial dataset with our model
    data_loader.shuffle = False
    d_size = len(data_loader.dataset)
    start_idx = 0


    gen_opt = optim.Adam(attacker.G.parameters(), lr=args.lr,
                             betas=(args.momentum, .99))

    for batch_idx, (data, target) in enumerate(data_loader):
        x, target = data.to(args.dev), target.to(args.dev)
        x_prime = attacker.batch_update(args, x,target, gen_opt,1)
        batch_size = len(data)
        end_idx = min(start_idx+batch_size, d_size)
        # Update batch in dataloader
        data_loader.dataset.data[start_idx:end_idx] = x_prime.squeeze()
        start_idx = end_idx

def baseline_transfer(args, attacker, attack_name, model_type, adv_img_list,
        list_classifiers=[], loaded_classifiers=None):
    print("Testing adversarial dataset from %s on %s classifiers" % (str(attack_name), model_type))
    total_fool_rate, i = [], 0
    if len(list_classifiers) == 0:
        list_classifiers = glob.glob(os.path.join(args.dir_test_models, "pretrained_classifiers", args.dataset, model_type, "model_*.pt"))

    for i, test_classif_path in enumerate(list_classifiers):
        if loaded_classifiers is None:
            if model_type == 'Ensemble Adversarial':
                classifier = load_model(args, type=args.type, filename=test_classif_path).to(args.dev)
            else:
                classifier = load_unk_model(args, test_classif_path, name=model_type)
        else:
            classifier = loaded_classifiers[i]
        classifier.eval()
        total_fool_rate.append(baseline_eval_classifier(args, attacker,
                                                    attack_name, adv_img_list,
                                                    classifier, model_type, i))
        del classifier
        torch.cuda.empty_cache()
        i = i + 1

    avg_fool_rate = np.mean(total_fool_rate)
    std_fool_rate = np.std(total_fool_rate)
    for j in range(0, len(total_fool_rate)):
        print("%s | Model %s number %d | Test Fool Rate %f" %(attack_name, model_type, j ,
                                              total_fool_rate[j]))
        if args.wandb:
            model_name = "Model " + model_type + " number " + str(j) + " Fool Rate"
            wandb.log({model_name: total_fool_rate[j]}, commit=False)

    print("Avg Fool Rate for %s is %f, STD: %f" %(attack_name,
                                                          avg_fool_rate,
                                                          std_fool_rate))
    if args.wandb:
        wandb.log({"Avg. Test Fool Rate on %s"%model_type: avg_fool_rate,
                    "Std Test Fool Rate on %s"%model_type: std_fool_rate,
                   "Indiv. Fool Rates %s"% model_type: total_fool_rate})
    return avg_fool_rate

def transfer(args, attacker, test_loader=None, list_classifiers=[]):
    # Create an adversarial dataset to test on other classifiers
    print("Creating an adversarial dataset with attack model")
    adv_loader = adv_data(args, attacker, test_loader)

    print("Testing adversarial dataset on other classifiers")
    for name, classifier in list_classifiers.items():
        eval_classifier(args, attacker, test_loader, classifier, name, adv_loader)

def baseline_eval_classifier(args, attacker, attack_name, test_loader,
                             classifier, name, i):
    correct_clean = 0
    correct_adv = 0
    avg_perturb_loss_list = []
    num_samples = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if not isinstance(target, torch.Tensor) or target.dim()==0:
            target = torch.tensor([target])

        x, target = data.to(args.dev), target.to(args.dev)
        x = x.clamp(0, 1)
        pred_adv = classifier(x.detach())
        out_adv = pred_adv.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_adv_tensor = out_adv.eq(target.unsqueeze(1).data)
        correct_adv += correct_adv_tensor.sum()
        num_samples += len(x)

    fool_rate = 100. * (1 - correct_adv / float(num_samples))
    # LOGGING
    # print("\n Attack Name: {} Name: {}_{} Transfer Fool Rate: {:.4f})".format(attack_name, name, i, fool_rate))
    return fool_rate.detach().cpu().numpy()

def eval_classifier(args, attacker, test_loader, classifier, name, adv_loader=None):
    """
    Evaluate adversarial samples from attacker on classifier. If `adv_loader`
    is provided, will use adversarial samples from this loader instead of the
    attacker
    """
    test_loader.shuffle = False
    if adv_loader is not None:
        adv_loader.shuffle = False
        adv_iterator = iter(adv_loader)

    print("Evaluating on %s" % name)
    classifier.to(args.dev)
    # EVALUATION
    test_itr = tqdm(enumerate(test_loader),
                    total=len(list(test_loader)))
    correct_clean = 0
    correct_adv = 0
    avg_perturb_loss_list = []
    perturb_loss_func = kwargs_perturb_loss[args.perturb_loss]

    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.dev), target.to(args.dev)
        # Get adv samples from model or adversarial loader if available
        if adv_loader is not None:
            adv_inputs, _  = next(adv_iterator)
        elif args.not_use_labels:
            adv_inputs = attacker.perturb(x)
        else:
            adv_inputs = attacker.perturb(x, target=target)

        adv_inputs = torch.clamp(adv_inputs, min=.0, max=1.)
        dist_perturb = perturb_loss_func(x, adv_inputs)
        if torch.all(dist_perturb > args.epsilon + 1e-6):
            import ipdb;ipdb.set_trace()
        assert torch.all(dist_perturb <= args.epsilon + 1e-6), (
               f"The adv input is not in the ball, "
               f"loss_pertub: {dist_perturb}"
               f"espilon {args.epsilon}")

        avg_perturb_loss_list.append(dist_perturb.mean().item())

        if not (torch.all(adv_inputs <= 1.0) and torch.all(adv_inputs >= 0.)):
            import pdb; pdb.set_trace()

        pred = classifier(x.detach())
        pred_adv = classifier(adv_inputs.detach())
        out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
        out_adv = pred_adv.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_adv_tensor = out_adv.eq(target.unsqueeze(1).data)
        correct_adv += correct_adv_tensor.sum()
        correct_clean += out.eq(target.unsqueeze(1).data).sum()

    clean_test_acc = 100. * correct_clean.cpu().numpy() / len(test_loader.dataset)
    adv_test_acc = 100. * correct_adv.cpu().numpy() / len(test_loader.dataset)

    # LOGGING
    print("\nTest set: Acc on Adversarial: {}/{} ({:.0f}%), ".format(correct_adv,
                                                           len(test_loader.dataset),
                                                           adv_test_acc))

    print("Test set: Acc on Clean: {}/{} ({:.0f}%)".format(correct_clean,
                                                   len(test_loader.dataset),
                                                   clean_test_acc))

    avg_perturb_loss = sum(avg_perturb_loss_list) / len(avg_perturb_loss_list)
    print(f"Test set Avg. Distortion: {avg_perturb_loss}, "
          f"Ball: {args.attack_ball}\n")

    if args.wandb:
        n_imgs = min(32,len(x))
        clean_image = (x)[:n_imgs].detach()
        adver_image = (adv_inputs)[:n_imgs].detach()
        delta_image = adver_image - clean_image
        file_base = "adv_images/test/" + args.namestr + "/"
        if not os.path.exists(file_base):
            os.makedirs(file_base)

        img2log_clean = save_image_to_wandb(args, clean_image, file_base+"clean.png", normalize=True)
        img2log_adver = save_image_to_wandb(args, adver_image, file_base+"adver.png", normalize=True)
        img2log_delta = save_image_to_wandb(args, delta_image, file_base+"delta.png", normalize=True)
        wandb.log({"%s/Test Adv Accuracy"%name: adv_test_acc, "%s/Test Clean Accuracy"%name: clean_test_acc,
                   '%s/Test Clean image'%name: [wandb.Image(img, caption="Clean") for img in img2log_clean],
                   '%s/Test Adver_image'%name: [wandb.Image(img, caption="Adv, "f"Label: {target[i]}"
                                                    f" Predicted: {out_adv[i].item()}")
                                        for i, img in enumerate(img2log_adver)],
                   '%s/Test Delta_image'%name: [wandb.Image(img, caption="Delta") for img in img2log_delta]}
                  )

    classifier.cpu()
    return adv_test_acc
