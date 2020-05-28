def GenerateCaptionsWithBeamSearchValueScoring(features, captions, model, beamSize=5):
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    candidates = [(gen_caps, 0)]
    for t in range(max_seq_len-1):
        next_candidates = []
        for c in range(len(candidates)):
            output = model(features, candidates[c][0])
            probs, words = torch.topk(output[:, -1:, :], beamSize)
            for i in range(beamSize):
                cap = torch.cat((candidates[c][0], words[:, :, i]), axis=1)
                value = valueNet(features.squeeze(0), cap).detach()
                score = candidates[c][1] -0.6*value.item() - 0.4*torch.log(probs[0, 0, i]).item()
                next_candidates.append((cap, score))
        ordered_candidates = sorted(next_candidates, key=lambda tup:tup[1])
        candidates = ordered_candidates[:beamSize]
    return candidates


with torch.no_grad():
    max_seq_len = 17
    captions, features, urls = sample_coco_minibatch(small_data, batch_size=100, split='train')
    for i in range(5):
        gen_caps = []
        gen_caps.append(GenerateCaptionsWithBeamSearchValueScoring(features[i:i+1], captions[i:i+1], policyNet)[0][0][0])
        decoded_tru_caps = decode_captions(captions[i], data["idx_to_word"])

      try:
            plt.imshow(image_from_url(urls[i]))
            plt.show()
        except:
            continue
        print(urls[i])
        print("Reinforcement learning: ",decode_captions(gen_caps[2], data["idx_to_word"]))
      
