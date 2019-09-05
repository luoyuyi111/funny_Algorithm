#pragma once
struct List{
	int value;
	List *next;
	int length;
	List *pre;
}
class LinkList{
public:
	LinkList();
	~ LinkList();
	void insertdata1(int u);
	void insertdata2(int u);
	void updatedata(int position,int u);
	void deletedata(int position);
	int lengthdata();
	int selectdata(int position);
}
