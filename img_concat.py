import cv2

if __name__ == '__main__':
    exps = ['Hartmann6', 'StyblinskiTang8', 'FiveWellPotentioal']

    imgs = [cv2.imread(f'exp_result/{exp}/{exp}_performance.png') for exp in exps]
    img_all = cv2.hconcat(imgs)
    cv2.imwrite('exp_result/all_continuous.png', img_all)
