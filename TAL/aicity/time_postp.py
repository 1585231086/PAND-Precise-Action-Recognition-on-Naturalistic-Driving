import argparse
import json
import os
import csv
import numpy as np
import shutil

def rmdir(pdir):
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        print("{} path has been delete.".format(pdir))
    else:
        print('{} column does not exist.'.format(pdir))

def mkdir(pdir):
    if os.path.isdir(pdir):
        print("{} path has existed.".format(pdir))
    else:
        os.makedirs(pdir)
        print('{} path has been built.'.format(pdir))
# Global parameters
AICITY_DATA_ROOT = '/xxxx/AICity'  # TODO: change to your own data root path.

def IOU(s1, e1, s2, e2):
    """
    Calculate IoU of two proposals
    :param s1: starting point of A proposal
    :param e1: ending point of A proposal
    :param s2: starting point of B proposal
    :param e2: ending point of B proposal
    :return: IoU value
    """
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def softNMS(video,f,det_file,link_file,dash,first):
    """
    soft-NMS for all proposals
    :param df: input dataframe
    :return: dataframe after soft-NMS
    """

    datas = np.loadtxt(f)
    lens=int(len(datas)/2)


    tstart, tend, tscore,tlabel = [], [], [],[]
    for i in range(len(video)):
        if video[i]['segment'][1]<lens:
            tstart.append(video[i]['segment'][0])
            tend.append(video[i]['segment'][1])
            tscore.append(video[i]['score'])
            tlabel.append(video[i]['label'])
    if not first:
        file =open(det_file)
        line = file.readline()
        while line:
            line = file.readline()
            item=line.split(' ')
            if item[0]==dash:
                tstart.append(int(item[2])-1)
                tend.append(int(item[3]))
                tscore.append(0.98)
                tlabel.append(item[2])

        filepath=open(link_file)
        line = filepath.readline()
        while line:
            line = filepath.readline()
            item=line.split(' ')
            if item[0]== dash :
                tstart.append(int(item[2]))
                tend.append(int(item[3]))
                tscore.append(0.87)
                tlabel.append(item[1])

    list = []
    means=np.mean(datas,axis=0)
    for i in range(len(datas)):
        if i%2==0:
            if (datas[i][1])>means[1]:
                list.append(1)
            else:
                list.append(0)


    for i in range(len(list)):
        if list[i]==0:
            i=i+1
        else:
            k=i
            while list[k]==1 and k<len(list)-1:
                k=k+1
            j=k
            while list[j]==0 and j<len(list)-1:
                j=j+1
            if i > 400 and i<500:
                print('jk',j,k)
                # print()
            if j-k>10 and j-k<34:

                tstart.append(k+1)
                tend.append(j-1)
                tscore.append(0.85)
                tlabel.append('0')
                tstart.append(k+6)
                tend.append(j-1)
                tscore.append(0.85)
                tlabel.append('0')
                tstart.append(k+1)
                tend.append(j-7)
                tscore.append(0.85)
                tlabel.append('0')
                tstart.append(k+1)
                tend.append(j-1)
                tscore.append(0.85)
                tlabel.append('0')
            i=j-2
            # print('i',i)
    start_time=14
    end_time=1
    if len(list)>610:
        start_time=75
        end_time=2

    rstart, rend, rscore ,rlabel= [], [], [],[]
    while len(tscore) > 1 and len(rscore) < top_number:
        while len(tscore) > 1 and len(rscore) < top_number:
            max_index = tscore.index(max(tscore))
            tmp_start = tstart[max_index]
            tmp_end = tend[max_index]
            tmp_score = tscore[max_index]
            tmp_label=tlabel[max_index]
            tmp_width = tmp_end - tmp_start
            maxx=max(tend)
            if tmp_width<10 or tmp_start<14 or maxx-tmp_end<1:
                # print(tmp_width)
                tstart.pop(max_index)
                tend.pop(max_index)
                tscore.pop(max_index)
                tlabel.pop(max_index)
            else:
                break
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        rlabel.append(tmp_label)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)
        tlabel=np.array(tlabel)

        tt1 = np.maximum(tmp_start-5, tstart)
        tt2 = np.minimum(tmp_end+5, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(np.float)

        idxs = np.where(intersection>0)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / 0.75)

        tstart = remove_(tstart,idxs)
        tend = remove_(tend,idxs)
        tscore = remove_(tscore,idxs)
        tlabel = remove_(tlabel,idxs)

    # print(rscore)
    post = []
    for j in range(len(rscore)):
        post.append([rscore[j], rstart[j], rend[j],rlabel[j]])

    return post

def remove_(a,b):
    a_index = [i for i in range(len(a))]
    a_index = set(a_index)
    b_index = set(b)
    index = list(a_index - b_index)
    a = [a[i] for i in index]
    return a


def sub_processor(raw_file, out_file,f,dash,rear,right,det_file,link_file,first):
    """
    Define job for every subprocess
    :param lock: threading lock
    :param pid: sub processor id
    :param video_list: video list assigned to each subprocess
    :return: None
    """
    # text = 'processor %d' % pid
    # with lock:
    #     progress = tqdm.tqdm(
    #         total=len(video_list),
    #         position=pid,
    #         desc=text
    #     )
    data = json.load(open(raw_file, 'r'))
    dict_ = []

    #output json file
    datas=data['results'][dash]
    datas.extend(data['results'][rear])
    datas.extend(data['results'][right])
    # for i in range(len(datas)):
    video = datas
    post_video = softNMS(video,f,det_file,link_file,dash,first)
    dict_.append(post_video)
    viz(dict_[0],f,out_file)
    print()


def viz(dict_,f,out_file):
    '''
    post-processing and visualize
    :param dict_:proposals
    :param f: file path
    :param out_file: output file path
    :return:
    '''
    import matplotlib.pyplot as plt
    import csv
    #raw proposals
    out=[]
    for result in dict_:
        strat=result[1]
        end=result[2]
        out.append(([[strat,end],[0.05,0.05],'r-']))
    datas = np.loadtxt(f)


    list = []
    means=np.mean(datas,axis=0)
    # print(means[1])
    for i in range(len(datas)):
        # list[i]=datas[i].split(' ')[0]
        if i%2==0:
            if (datas[i][1])>means[1]:
                list.append(1)
            else:
                list.append(0)

    convert=[]
    datas=datas.tolist()
    xpre=datas[0].index(max(datas[0]))
    # print(max(datas[0]))
    for i in range(len(datas)):
        # list[i]=datas[i].split(' ')[0]
        if i%2==0:
            dix=datas[i][1:].index(max(datas[i][1:]))
            score=datas[i][dix+1]
            if dix !=xpre and score>0.8:
                convert.append(1)
            else:
                convert.append(0)
            xpre=dix
    y = [i for i in range(int(len(datas)/2)+1)]


    dict_.sort(key = lambda x:x[1], reverse=False)
    dict=dict_
    i=0
    st_pro=[]

    last=0
    while i <(len(list)):
        if list[i]==0:
            i=i+1
        else:
            k=i
            while list[k]==1:
                k+=1
            if k-i<5:
                con_indx=i
                while convert[con_indx]==1:
                    con_indx-=1
                if k-con_indx>7:
                    st_pro.append([k + 1, 0])
                elif k-i>2:
                    j = k
                    while list[j + 1] == 0 and j<len(dict):
                        j += 1
                    if j-k>11 :
                        st_pro.append([k + 2, 0])
                i=k
                last=k
            else:
                j=k

                while list[j+1]==0 and j+1<len(dict):
                    j+=1
                if i-last>7:
                    st_pro.append([i-2, 1])
                if j-k>7 :
                    st_pro.append([k+1,0])
                i=j
                last=k


    ref=[]
    prev=0
    for i in range(len(list)):
        a= bool(list[i]) or bool(convert[i])
        if a !=prev:
            ref.append(1)
        else:
            ref.append(0)
        prev=a


    dicts=fine_tune(dict,st_pro,list,convert,ref,out_file)

    for result in dicts:
        strat=result[1]
        end=result[2]
        out.append(([[strat,end],[0.04,0.04],'b-']))

    for item in out:
        plt.plot(item[0],item[1],item[2])

    view_list=[]
    for ind in list:
        if ind ==0:
            view_list.append(0)
        else:view_list.append(0.1)

    plt.bar(y, view_list)


    # plt.rcParams['figure.figsize'] = (8.0, 2.0)
    ax_val=[0, 550,0,0.1]
    plt.axis(ax_val)
    # plt.rcParams['figure.figsize'] = (20.0, 2.0)
    # plt.figure(figsize=(8, 2))
    plt.gcf().set_size_inches(20, 4)
    # plt.savefig("t1.png",bbox_inches ='tight')
    # plt.savefig(‘test.png’, bbox_inches =“tight”)
    # plt.show()


# def deal_prop(dict,list,convert,idx):

def fine_tune(dict,st_pro,list,convert,ref,out_file):
    '''
    :param dict: proposals
    :param st_pro:
    :param list: if is background
    :param convert: if type is change
    :param ref: if boundary
    :param out_file:
    :return:proposals
    '''
    sta_in=0
    lists=[]
    for item in st_pro:
        lists.append(item[0])
    for item in dict:
        if item[0]>0.95:
            lists.append(item[1])
            lists.append(item[2])
    for ind in st_pro:
        max=99
        index=0
        for i in range(len(dict)):
            if abs(dict[i][ind[1]+1]-ind[0])<max:
                index=i
                max=abs(dict[i][ind[1]+1]-ind[0])
        # print(ind)
        if ind[1]==1 and dict[index][0]<0.9:
            if ind[0]-dict[index][1]<14:
                offset=dict[index][2]-ind[0]
                dict[index][2]=dict[index][2]-offset
                dict[index][1]=dict[index][1]-offset
                sr_ind=index
                if index!=0:
                    while dict[sr_ind][1]-dict[sr_ind-1][2]<7 and sta_in<sr_ind:
                        sr_ind-=1
                        dict[sr_ind][2] = dict[sr_ind][2] - offset
                        dict[sr_ind][1] = dict[sr_ind][1] - offset
            else:
                dict[index][2]=ind[0]
            sta_in=index
        if ind[1]==0 and dict[index][0]<0.9:
            dict[index][1]=ind[0]
            sta_in=index
    for ind in range(len(dict)):
        if dict[ind][0]==0.98:
            flag=0
            for i in range(int(dict[ind][1]),int(dict[ind][2])):
                if list[i]==1: flag=1
            if flag==0:
                strat = int(dict[ind][1])
                end = int(dict[ind][2])
                if list[strat] == 0:
                    i = strat
                    while list[i] == 0 and strat - i < 10:
                        i -= 1
                    if list[i] != 0:
                        dict[ind][1] = i+3
                if list[end] == 0:
                    k = end
                    while list[k] == 0 and k - end < 7:
                        k += 1
                    if  list[k] != 0:
                        dict[ind][2] = k-3


    for ind in range(len(dict)):
        if dict[ind][0]==0.85:
            strat=int(dict[ind][1])
            end=int(dict[ind][2])
            if ref[end]==0 :
                i=end
                while ref[i] == 0   and end-i<3 :
                    i -= 1
                k=end
                while ref[k] == 0 and  k -end< 3:
                    k += 1
                if end-i <k-end and ref[i]!=0:
                    dict[ind][2]=i
                elif end-i >=k-end and ref[k]!=0:
                    dict[ind][2] = k-1




    for ind in range(len(dict)-1):
        interval=dict[ind+1][1]-dict[ind][2]
        if interval>10 and ind<len(dict)-1 :
            if dict[ind+1][0]>0.85 and dict[ind][0]<0.84:

                i=int(dict[ind][2])+1
                while list[i]==0 and convert[i]==0:
                    i+=1
                if i-dict[ind][1]<26:
                    dict[ind][2]=i-1
            elif dict[ind+1][0]<0.8 and dict[ind][0]>0.84:
                if dict[ind+1][2]-dict[ind][2]-9<28:
                    dict[ind+1][1]=dict[ind][2]+9
            elif dict[ind+1][0]<0.8 and dict[ind][0]<0.8:
                dict[ind][2] = dict[ind+1][1] - 10


    k=-1
    p=-1

    for ind in range(len(dict) - 1):
        interval = dict[ind + 1][1] - dict[ind][2]
        if interval>10 and ind<len(dict)-1 :
                if dict[ind+1][1]-dict[ind][1]-9>35 and int(dict[ind+1][1])-16-int(dict[ind][2])>12:
                    dict.append([0.9,int(dict[ind][2])+8,int(dict[ind+1][1])-8,'0'])
        lens=int(dict[ind][2])-int(dict[ind][1])
        if lens >45:
            start=dict[ind][1]+(int(lens))/2
            end=dict[ind][2]
            dict[ind][2]=start
            if dict[0][0]<dict[len(dict)-1][0]:
                k=0
            else:
                k=17
            dict.append([0.9,start+3,end,'0'])

    if k!=-1:dict.pop(k)


    file = open(out_file, 'w')
    for i in range(len(dict)):

        s = str(dict[i][1:]).replace('{', '').replace('}', '').replace("'", '').replace(':', ' ').replace('[', '').replace(']', '') + '\n'

        file.write(s)
    file.close()


    return dict


if __name__ == "__main__":
    """ Define parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'E:\s_program\data_pocc\thumos14_1_rgb_3.json',
                        help='TAL model output json file')
    parser.add_argument('--output_dir', type=str, default=r'output', help='output proposals root')
    parser.add_argument('--recog_result_dir', type=str,
                        default=r'data_result/val49831/Dashboard_User_id_49381_1_latest_results_4.txt',
                        help='Action recognition reuslts root')
    parser.add_argument('--video_ids', type=str, default='video_ids.csv', help='video ids file')
    parser.add_argument('--det_file', type=str, default=r'results_submission_second_link123.txt',
                        help='detection results')
    parser.add_argument('--link_file', type=str, default=r'results_submission_second_link.txt',
                        help='action recognition post-processing reuslt ')
    parser.add_argument('--top_number', type=int, nargs='?', default=18)
    parser.add_argument('--first', type=bool, default=True)
    args = parser.parse_args()

    """ Number of proposal needed to keep for every video"""
    top_number = args.top_number
    """ Number of thread for post processing"""
    # thread_num = args.thread
    post = ['_swin_dashboard_results_4', '_swin_rearview_results_4', '_swin_rightside_results_4']
    videos_info = []
    with open(args.video_ids, 'r') as f: # TODO video_ids.cv
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            videos_info.append(row)
    for i, multi_videos in enumerate(videos_info):
        mkdir(args.output_dir)
        output_file = os.path.join(args.recog_result_dir,'stamp_results', 'bbox1', multi_videos[1].split('.MP4')[0]+'.txt')
        mkdir(os.path.split(output_file)[0])
        result_file = os.path.join(args.recog_result_dir, multi_videos[1].split('.')[0])+post[0]+'.txt'
        sub_processor(args.input_dir, output_file, result_file, multi_videos[1].split('.MP4')[0],
        multi_videos[2].split('.MP4')[0], multi_videos[3].split('.MP4')[0], args.det_file, args.link_file,args.first)
