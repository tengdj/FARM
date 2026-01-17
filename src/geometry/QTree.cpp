#include "MyPolygon.h"



/*
 * functions for QTree
 * */

void QTNode::split(){
	double mid_x = (mbr.high[0]+mbr.low[0])/2;
	double mid_y = (mbr.high[1]+mbr.low[1])/2;
	children[bottom_left] = new QTNode(mbr.low[0],mbr.low[1],mid_x,mid_y);
	children[bottom_right] = new QTNode(mid_x,mbr.low[1],mbr.high[0],mid_y);
	children[top_left] = new QTNode(mbr.low[0],mid_y,mid_x,mbr.high[1]);
	children[top_right] = new QTNode(mid_x,mid_y,mbr.high[0],mbr.high[1]);
	for(int i=0;i<4;i++){
		children[i]->level = level+1;
	}
	for(MyPolygon *obj:objects){
		Point p = obj->getMBB()->centroid();
		int dim = 2*(p.y>=mid_y)+(p.x>=mid_x);
		assert(dim>=0 && dim <=3);
		children[dim]->objects.push_back(obj);
	}
	objects.clear();
}

void QTNode::merge(){
	if(isleaf()){
		return;
	}
	for(int i=0;i<4;i++){
		children[i]->merge();
		objects.insert(objects.end(), children[i]->objects.begin(), children[i]->objects.end());
		delete children[i];
		children[i] = NULL;
	}
	assert(isleaf());
}

void QTNode::touch(Point &p, MyPolygon *obj){
	if(!mbr.contain(p)){
		return;
	}
	if(!isleaf()){
		int offset =  2*(p.y>(mbr.low[1]+mbr.high[1])/2)+(p.x>(mbr.low[0]+mbr.high[0])/2);
		assert(offset<4 && offset>=0);
		assert(children[offset]);
		children[offset]->touch(p, obj);
	}else{
		lock();
		objects.push_back(obj);
		unlock();
	}
}

void QTNode::adjust(const size_t threshold){
	if(isleaf()){
		if(objects.size()>threshold){
			split();
			for(int i=0;i<4;i++){
				children[i]->adjust(threshold);
			}
		}
	}else{
		int leafchild = 0;
		size_t onum = 0;
		for(int i=0;i<4;i++){
			children[i]->adjust(threshold);
			if(children[i]->isleaf()){
				onum += children[i]->objects.size();
				leafchild ++;
			}
		}
		// oversplitted, merge the node to leaf
		if(leafchild==4 && onum <= threshold){
			merge();
		}
	}
}

