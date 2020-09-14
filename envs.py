"""D2D edge computing offloading decision optimization environment
version.1.1 by Kim Yeong-Je 2020.02.02"""

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import os
import torch
import random

class D2DEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):
        self.num_UE=6
        self.observation_space=spaces.Box(low=0,high=12,shape=[15],dtype=np.float32)
        #observation_space는 user가 알아야하는 정보들을 가지고 있어야함.
        #ex)last_task_size,task_intensity,task_state,task_location,left_deadline...
        self._action_set=[[0],[1],[2],[3],[4],[5]]
        self.action_space=spaces.Discrete(len(self._action_set))
        #action_space는 기존에 생각한 데로 0부터 n까지의 숫자로 정하자.
        #0이면 working,1-n이면 offloading하는 것으로 생각해볼 수 있다. 어떤 UE로 offloading될지는 UE+action%n 으로 결정할 수 있음.
        
    def reset(self):
        self.state=np.array([])        
        self.done=False
        self.time=0
        #self.simulation_limit=0
        #self.energy_consumption=0
        self.reset_UE()
        self.UE_set[0].reset_task()
        for i in range(self.num_UE):
            self.UE_set[i].make_queue()
        self.UE_set[0].reset_transmission_list()
        self.UE_set[0].make_transmission_list()
        self.num_of_iter=0
        self.simul_time=0
        self.init_action=0
        self.reward=0
        self.action_factor=0
        return1=[0,0,0,0,0,0]
        return2=[0,0,0,0,0,0]
        return3=[0,0,0,0,0,0]
        
        for i in range(6):
            if self.UE_set[i].task_list.size==0:
                return1[i]=0
                return2[i]=0
            else:
                return1[i]=self.UE_set[i].task_list[1][0]/1000
                return2[i]=self.UE_set[i].task_list[2][0]
                return3[i]=self.UE_set[i].task_list.size/4
        return4=[0,0,0]
        
        if self.UE_set[0].transmission_set.size==0:
            return4=[0,0,0]
        else:
            return4=[self.UE_set[0].transmission_set[2][0],self.UE_set[0].transmission_set.size/6,self.UE_set[0].transmission_set[5][0]]
            
        self.action_factor=0
        if self.UE_set[0].transmission_set.size!=0:
            self.action_factor=6-self.UE_set[0].transmission_set[5][0]
        return3.extend(return1)        
        return3.extend(return4)
        #state=[self.simul_time,return1[0],return1[1],return1[2],return1[3],return1[4],return1[5],return2[0],return2[1],return2[2],return2[3],return2[4],return2[5],return3[0],return3[1],return3[2],return3[3],return3[4],return3[5],return4[0],return4[1],return4[2],return4[3],return4[4],self.UE_set[0].UE_channel[0][1],self.UE_set[0].UE_channel[0][2],self.UE_set[0].UE_channel[0][3],self.UE_set[0].UE_channel[0][4],self.UE_set[0].UE_channel[0][5],self.UE_set[0].UE_channel[1][0],self.UE_set[0].UE_channel[1][2],self.UE_set[0].UE_channel[1][3],self.UE_set[0].UE_channel[1][4],self.UE_set[0].UE_channel[1][5],self.UE_set[0].UE_channel[2][0],self.UE_set[0].UE_channel[2][1],self.UE_set[0].UE_channel[2][3],self.UE_set[0].UE_channel[2][4],self.UE_set[0].UE_channel[2][5],self.UE_set[0].UE_channel[3][0],self.UE_set[0].UE_channel[3][1],self.UE_set[0].UE_channel[3][2],self.UE_set[0].UE_channel[3][4],self.UE_set[0].UE_channel[3][5],self.UE_set[0].UE_channel[4][0],self.UE_set[0].UE_channel[4][1],self.UE_set[0].UE_channel[4][2],self.UE_set[0].UE_channel[4][3],self.UE_set[0].UE_channel[4][5],self.UE_set[0].UE_channel[5][0],self.UE_set[0].UE_channel[5][1],self.UE_set[0].UE_channel[5][2],self.UE_set[0].UE_channel[5][3],self.UE_set[0].UE_channel[5][4]]
        state=np.array(return3)
        return state
        #channel을 nxn list로 줘야하는데 reset할때마다 channel을 결정할지, action마다 channel을 결정할지.(상의를 해보고 결정)
    
    def step(self,action):
        """action이 어디서 들어가야할지 생각해야함 + done은 어떻게 만들지?"""
        """외부 loop의 break가 done 여부를 결정하는 지점들임."""
        self.UE_set[0].reset_signal()
        for i in range(self.num_UE):
            self.UE_set[i].idle_time=self.simul_time-self.UE_set[i].work_time
        for cur_time in range(self.simul_time,10000):
            if cur_time%20==0:
                self.UE_set[0].reset_channel()
            self.num_of_iter+=1
            for i in range(self.num_UE):
                self.UE_set[i].signal_gen(cur_time)
                self.UE_set[i].execution(cur_time)
                
            if self.UE_set[0].signal is True:
                #print('simulation break')
                self.simul_time=cur_time
                break
            elif self.UE_set[0].task_list.size==0 and self.UE_set[1].task_list.size==0 and self.UE_set[2].task_list.size==0 and self.UE_set[3].task_list.size==0 and self.UE_set[4].task_list.size==0 and self.UE_set[5].task_list.size==0 and self.UE_set[0].transmission_set.size==0:
                self.simul_time=cur_time
                break
        for i in range(self.num_UE):
            self.UE_set[i].transmission(action)
            if self.UE_set[0].transmission_signal==True:
                self.UE_set[0].reset_trans_signal()
                break
        self.num_of_iter-=1
        if self.num_of_iter>=9999:
            reward=-5000-(self.UE_set[1].work_time+self.UE_set[2].work_time+self.UE_set[3].work_time+self.UE_set[4].work_time+self.UE_set[5].work_time)*0.15-(self.UE_set[1].idle_time+self.UE_set[2].idle_time+self.UE_set[3].idle_time+self.UE_set[4].idle_time+self.UE_set[5].idle_time)*0.1
            reward=int(reward)
            self.simul_time=10000
            self.done=True
        elif self.UE_set[0].task_list.size==0 and self.UE_set[1].task_list.size==0 and self.UE_set[2].task_list.size==0 and self.UE_set[3].task_list.size==0 and self.UE_set[4].task_list.size==0 and self.UE_set[5].task_list.size==0 and self.UE_set[0].transmission_set.size==0:
            reward=10000+(10000-self.simul_time)-(self.UE_set[1].work_time+self.UE_set[2].work_time+self.UE_set[3].work_time+self.UE_set[4].work_time+self.UE_set[5].work_time)*0.15-(self.UE_set[1].idle_time+self.UE_set[2].idle_time+self.UE_set[3].idle_time+self.UE_set[4].idle_time+self.UE_set[5].idle_time)*0.1
            reward=int(reward)
            self.done=True 
        else:
            reward=0
            self.done=False
        if self.UE_set[0].transmission_set.size==0:
            pass
        
        if self.done==True:
            #print(reward)
            pass
            
        return1=[0,0,0,0,0,0]
        return2=[0,0,0,0,0,0]
        return3=[0,0,0,0,0,0]
        
        for i in range(6):
            if self.UE_set[i].task_list.size==0:
                return1[i]=0
                return2[i]=0
            else:
                return1[i]=self.UE_set[i].task_list[1][0]/1000
                return2[i]=self.UE_set[i].task_list[2][0]
            return3[i]=self.UE_set[i].task_list.size/4
        return4=[0,0,0]
        
        if self.UE_set[0].transmission_set.size==0:
            return4=[0,0,0]
        else:
            return4=[self.UE_set[0].transmission_set[2][0],self.UE_set[0].transmission_set.size/6,self.UE_set[0].transmission_set[5][0]]
        """transmission_set 구성요소: task_number, cycle, start_time, deadline, data_size, location"""
        """전체 task를 받아서 각각의 task의 location에 따라서 각각의 UE가 task_list를 만들도록 한다. 
        이 때 task_list는 위에서부터 task_number, task_cycle, task_gen_time, task_deadline이다."""
        """starttime, queue size, location만 넣으면됨."""
        self.action_factor=0
        if self.UE_set[0].transmission_set.size!=0:
            self.action_factor=6-self.UE_set[0].transmission_set[5][0]
                
        #self.state=[self.simul_time,return1[0],return1[1],return1[2],return1[3],return1[4],return1[5],return2[0],return2[1],return2[2],return2[3],return2[4],return2[5],return3[0],return3[1],return3[2],return3[3],return3[4],return3[5],return4[0],return4[1],return4[2],return4[3],return4[4],self.UE_set[0].UE_channel[0][1],self.UE_set[0].UE_channel[0][2],self.UE_set[0].UE_channel[0][3],self.UE_set[0].UE_channel[0][4],self.UE_set[0].UE_channel[0][5],self.UE_set[0].UE_channel[1][0],self.UE_set[0].UE_channel[1][2],self.UE_set[0].UE_channel[1][3],self.UE_set[0].UE_channel[1][4],self.UE_set[0].UE_channel[1][5],self.UE_set[0].UE_channel[2][0],self.UE_set[0].UE_channel[2][1],self.UE_set[0].UE_channel[2][3],self.UE_set[0].UE_channel[2][4],self.UE_set[0].UE_channel[2][5],self.UE_set[0].UE_channel[3][0],self.UE_set[0].UE_channel[3][1],self.UE_set[0].UE_channel[3][2],self.UE_set[0].UE_channel[3][4],self.UE_set[0].UE_channel[3][5],self.UE_set[0].UE_channel[4][0],self.UE_set[0].UE_channel[4][1],self.UE_set[0].UE_channel[4][2],self.UE_set[0].UE_channel[4][3],self.UE_set[0].UE_channel[4][5],self.UE_set[0].UE_channel[5][0],self.UE_set[0].UE_channel[5][1],self.UE_set[0].UE_channel[5][2],self.UE_set[0].UE_channel[5][3],self.UE_set[0].UE_channel[5][4]]
        return3.extend(return1) 
        return3.extend(return4)
        self.state=np.array(return3)
        
        self.reward=reward
        """state를 어떻게 만들어서 반환해야 잘 학습이 될지 생각해야한다. 각 UE가 현재 처리하고 있는 task상태?"""
        """또한 현재의 채널상태도 보내주는게 좋을 수 있다. 현재시간도 얼마인지 알아야 한다."""
        """지금 가진 data를 한번에 다 보내는 것과 선별적으로 보내는것의 차이가 무엇일지."""
        return self.state, reward, self.done, {}

    def reset_UE(self):
        self.UE_set=[]
        for i in range(self.num_UE):
            UE_tmp=UE(i)
            self.UE_set.append(UE_tmp)
    
    def seed(self, seed=None):
        pass
                

class task:
    def __init__(self,task_number):
        self.task_number=task_number
        self.start_time=np.random.randint(50,150)+100*self.task_number
        self.init_location=np.random.randint(1,6)
        #0이면 server,1-5이면 UE가 됨.
        #server에서 task가 생길 일은 없으므로 1-5사이에 생성시킴.
        #data_size는 Kbit단위, intensity는 cycle/bit단위, deadline은 sec단위
        self.curr_location=self.init_location
        self.local_data_size=np.random.randint(15000,17500)
        self.local_intensity=np.random.randint(100,150)
        #2.1Giga_cycle average
        self.local_cycle=self.local_data_size*self.local_intensity
        self.off_data_size=np.random.randint(450,500)
        self.off_intensity=np.random.randint(4000,5000)
        #2Giga_cycle average
        self.off_cycle=self.off_data_size*self.off_intensity
        self.deadline=1000
        self.local_done=False
        self.off_done=False

class channel:
    """channel의 data rate를 정의하기 위해서 만들어진 class이다.
    userM에서 userN까지의 data rate를 class로 만들어야함."""
    channel_array=np.array([])
    channel_factor=np.array([])
    def __init__(self):
        pass
    
    def make_array(UE_num):
        channel.channel_array=np.zeros((UE_num,UE_num))
        channel.rand=np.random.random((UE_num,UE_num))
        for j in range(UE_num):
            for i in range(UE_num):
                if i==j:
                    channel.channel_array[j][i]=0
                elif j==0:
                    if channel.rand[j][i]>0.2:
                        channel.channel_array[j][i]=np.random.randint(20,40)
                    else:
                        channel.channel_array[j][i]=np.random.randint(5,10)
                elif i>j:
                    if channel.rand[j][i]>0.05:
                        channel.channel_array[j][i]=np.random.randint(150,200)
                    else:
                        channel.channel_array[j][i]=np.random.randint(25,50)
        for j in range(UE_num):
            for i in range(UE_num):
                if j>i:
                    channel.channel_array[j][i]=channel.channel_array[i][j]
        
        return channel.channel_array        
        
        
class UE:
    #CPU_freq는 MHz단위
    signal=None
    sig_dic={}
    task_set=[task(0),task(1),task(2),task(3),task(4),task(5)]
    transmission_set=np.array([[0],[0],[0],[0],[0],[0]])
    UE_num=6
    UE_channel=channel.make_array(UE_num)
    transmission_signal=False
    transmission_count=0
    #UE class 변수로 task_set을 넣어두면 모든 UE들이 같은 task_set을 공유하게 만들 수 있음.
    
    def __init__(self,user_num):
        self.UE_num=user_num
        self.count=0
        self.idle_time=0
        self.work_time=0
        if user_num==0:
            self.CPU_freq=10000
            self.sigma=0
            #self.energy_factor=0
            #server=UE0
        else:
            self.CPU_freq=2000
            self.sigma=0.1
            #self.energy_factor=1
            #sigma는 self의 task를 위해 반드시 점유해야하는 CPU점유율이다.
            #local computing일 때는 sigma 만큼의 감산이 필요없고, offloading일 때는 (1-sigma)만큼의 감산이 필요하다.
        self.work=False
    
    def make_queue(self):
        """전체 task를 받아서 각각의 task의 location에 따라서 각각의 UE가 task_list를 만들도록 한다. 
        이 때 task_list는 위에서부터 task_number, task_cycle, task_gen_time, task_deadline이다."""
        self.task_list=np.array([[0],[0],[0],[0]])
        for i in range(len(UE.task_set)):
            if self.task_set[i].init_location==self.UE_num:
                #print('Task',UE.task_set[i].task_number, 'is in UE',self.UE_num)
                self.task_list=np.hstack((self.task_list,np.array([[UE.task_set[i].task_number],[UE.task_set[i].local_cycle],[UE.task_set[i].start_time],[UE.task_set[i].deadline]])))                
        self.task_list=np.delete(self.task_list,0,1)

    def make_transmission_list(self):
        for i in range(len(UE.task_set)):
            UE.transmission_set=np.hstack((self.transmission_set,np.array([[UE.task_set[i].task_number],[UE.task_set[i].off_cycle],[UE.task_set[i].start_time],[UE.task_set[i].deadline],[UE.task_set[i].off_data_size],[UE.task_set[i].curr_location]])))

        UE.transmission_set=np.delete(UE.transmission_set,0,1)
    
    def reset_transmission_list(self):
        UE.transmission_set=np.array([[0],[0],[0],[0],[0],[0]])
        
    def execution(self,time):
        if self.task_list.size!=0:
            """queue안에 task가 들어있음"""
            if self.task_list[2][0]<=time:
                """task가 발생하여 처리되기 시작함"""
                self.work=True
                self.task_list[1][0]-=self.CPU_freq
                self.work_time+=1
            else:
                self.work=False
                pass
            if self.task_list[1][0]<0:
                """task done"""
                #print(self.task_list[0][0],'th task is done at',time,'by user',self.UE_num)
                self.task_list=np.delete(self.task_list,0,1)
                """queue에서 task를 지우고 다음 task를 queue에 앞에 배치해줌"""
                """다음 task가 어떤지 알려주는 UE.sig_dic을 생성함."""
                if self.task_list.size!=0:
                    UE.sig_dic={'time':time,'UE_loc':self.UE_num,'task_num':self.task_list[0][0]}
                else:
                    UE.sig_dic={'time':time,'UE_loc':self.UE_num,'task_num':None}
                    #print('UE',self.UE_num,'executed every task at',time)
                    self.count+=1
                    self.work=False
                #만약 task가 지워지고 뒤에 list가 없으면 에러가 생김.
        else:
            if self.count==0:
                #print('no task in',self.UE_num)
                pass
            else:
                pass
            self.count+=1
            self.work=False
            #UE.signal=self.UE_num
            #UE.signal+=1
            #이 signal을 어떤 UE가 날렸는지를 알 수 있게 만드는게 가장 중요함.
            
    def time_check(self):
        pass
    
    def signal_gen(self,time):
        if UE.transmission_set.size!=0:
            if UE.transmission_set[5][0]==self.UE_num:
                if UE.transmission_set[2][0]<=time:
                    UE.signal=True
        else:
            pass
        
    
    def transmission(self,action):
        """transmission이 되는 동안에는 touch되면 안되는거 아닐까?
        signal이 생겨서 나오면 그때 state의 상태를 보고 다음 action을 결정하는게 순서가 맞음. 그런데 지금은 action이 먼저 들어가고 signal이 생김.
        action을 받아서 transmission_set의 가장 앞에 있는 queue를 offloading할지 아니면 execution을 위해서 task_set으로 보내줄지 결정함."""
        if UE.transmission_set.size!=0:
            if UE.transmission_set[5][0]==self.UE_num:
                """transmission_queue에 task가 들어있음."""
                if action != UE.transmission_set[5][0]:
                    if action==0:
                        UE.transmission_count+=10
                    else:
                        UE.transmission_count+=1
                    transmission_time=UE.transmission_set[4][0]/(UE.UE_channel[UE.transmission_set[5][0]][action]/100)
                    UE.transmission_set[2][0]=UE.transmission_set[2][0]+transmission_time
                    UE.transmission_set[5][0]=action
                    index=np.argsort(UE.transmission_set)[2]
                    UE.transmission_set=UE.transmission_set.T[index].T
                    #self.energy_consumption=
                elif action== UE.transmission_set[5][0]:
                    self.task_list=np.hstack((self.task_list,np.array([[UE.transmission_set[0][0]],[UE.transmission_set[1][0]],[UE.transmission_set[2][0]],[UE.transmission_set[3][0]]])))
                    UE.transmission_set=np.delete(UE.transmission_set,0,1)
                    UE.transmission_signal=True
                    index=np.argsort(self.task_list)[2]
                    self.task_list=self.task_list.T[index].T
            else:
                pass
                #서로 다른 instance에서 서로의 instance variable을 touch할 수 있는지?
        else:    
            pass
    
    def reset_signal(self):
        """signal을 바깥에서 수정하면 각각의 값의 binding이 풀려버린다. 따라서 함수를 통해 초기화 시켜야 그런 문제가 발생하지 않게된다."""
        UE.signal=False
    
    def reset_channel(self):
        UE.UE_channel=channel.make_array(6).astype('float64')
    
    def reset_trans_signal(self):
        UE.transmission_signal=False
    
    def reset_task(self):
        
        UE.task_set=[task(0),task(1),task(2),task(3),task(4),task(5)]
