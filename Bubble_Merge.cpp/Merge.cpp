#include<iostream>
#include<stdlib.h>
#include<omp.h>
using namespace std;


void mergesort(int a[],int i,int j);
void merge(int a[],int i1,int j1,int i2,int j2);

void mergesort(int a[],int i,int j)
{
    int mid;
    if(i<j)
    {
        mid=(i+j)/2;
        
        #pragma omp parallel sections 
        {

            #pragma omp section
            {
                mergesort(a,i,mid);        
            }

            #pragma omp section
            {
                mergesort(a,mid+1,j);    
            }
        }

        merge(a,i,mid,mid+1,j);    
    }

}
 
void merge(int a[],int i,int mid,int mid1,int j)
{
    int temp[1000];    
    int new_i,new_j,k;
    new_i=i;    
    new_j=mid1;    
    k=0;
    //sorting the element and compare both partition
    while(new_i<=mid && new_j<=j)    
    {
        if(a[new_i]<a[new_j])
        {
            temp[k++]=a[new_i++];
        }
        else
        {
            temp[k++]=a[new_j++];
	}    
    }
    // left partition remaining element merge into temp array 
    while(new_i<=mid)    
    {
        temp[k++]=a[new_i++];
    }
        //right partition remaining element merge into temp array 
    while(new_j<=j)    
    {
        temp[k++]=a[new_j++];
    }
        //all sorted element is merge in a array
    for(int loop_i=i,loop_j=0;loop_i<=j;loop_i++,loop_j++)
    {
        a[loop_i]=temp[loop_j];
    }    
}


int main()
{
    int *a,n,i;
    cout<<"\n enter total no of elements=>";
    cin>>n;
    a= new int[n];

    cout<<"\n enter elements=>\n";
    for(i=0;i<n;i++)
    {
        cin>>a[i];
    }
        
    mergesort(a, 0, n-1);
    
    cout<<"\n sorted array is=>";
    for(i=0;i<n;i++)
    {
        cout<<"\n"<<a[i];
    }
       
    return 0;
}

/*

guest-tim1wd@C04L0818:~$ g++ merge_sort.cpp -fopenmp
guest-tim1wd@C04L0818:~$ ./a.out

 enter total no of elements=>8

 enter elements=>
2
34
45
21
8
9
43
81

 sorted array is=>
2
8
9
21
34
43
45
81


*/
