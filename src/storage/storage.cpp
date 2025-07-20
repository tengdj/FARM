/*
 * storage.cpp
 *
 *  Created on: Jun 27, 2022
 *      Author: teng
 */

#include "Ideal.h"

std::mutex polygon_mtx;
std::queue<std::string> polygon_queue;
std::condition_variable polygon_cv;
bool polygon_done = false;

// 线程函数：从队列中读取WKT并解析到局部容器中
void process_lines(std::vector<Ideal*>& local_ideals) {
    while (true) {
        std::string line;
        {
            std::unique_lock<std::mutex> lock(polygon_mtx);
            polygon_cv.wait(lock, [] { return !polygon_queue.empty() || polygon_done; });

            if (polygon_queue.empty() && polygon_done)
                break;

            line = std::move(polygon_queue.front());
            polygon_queue.pop();
        }

        std::istringstream iss(line);
        std::string wkt;

        getline(iss, wkt); // 提取wkt
        if (wkt.empty()) {
            std::cerr << "read WKT failed\n";
            continue;
        }

        // 去掉引号
        // if (!wkt.empty() && wkt.front() == '"') {
        //     wkt.erase(0, 1); // 删除起始的引号
        // }
        // if (!wkt.empty() && wkt.back() == '"') {
        //     wkt.erase(wkt.size() - 1); // 删除结尾的引号
        // }

        size_t offset = wkt.find('(');
        local_ideals.push_back(read_polygon(wkt.c_str(), offset));
    }
}

std::vector<Ideal*> load_polygon_wkt(const char* path) {
    std::vector<Ideal*> ideals;

    if (!file_exist(path)) {
        log("%s does not exist", path);
        exit(1);
    }

    std::ifstream infile(path);
    if (!infile) {
        std::cerr << "can not open this file\n";
        exit(1);
    }

    std::vector<std::thread> workers;
    int num_threads = 1; // 获取系统支持的线程数量
	std::vector<std::vector<Ideal*>> local_ideals(num_threads); // 每个线程都有一个局部容器

    // 启动多个线程处理WKT
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(process_lines, std::ref(local_ideals[i]));
    }

    // 主线程读取文件并将每一行放入队列
    std::string line;
    while (getline(infile, line)) {
        std::unique_lock<std::mutex> lock(polygon_mtx);
        polygon_queue.push(std::move(line));
        polygon_cv.notify_one(); // 通知一个线程处理
    }

    infile.close();
    polygon_done = true;
    polygon_cv.notify_all(); // 通知所有线程处理结束

    // 等待所有线程结束
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

	for (const auto& local_vec : local_ideals) {
        ideals.insert(ideals.end(), local_vec.begin(), local_vec.end());
    }

	for(int i = 0; i < ideals.size(); i ++) ideals[i]->id = i; 

    return ideals;
}

Point* load_point_wkt(const char *path, size_t &count, query_context *ctx){
	vector<string> wkts;
	if(!file_exist(path)){
		log("%s does not exist",path);
		exit(1);
	}

	ifstream infile(path);
	if(!infile){
		std::cerr << "can not open this file\n";
		exit(1);
	}

	string line;
	// getline(infile, line);
	while(getline(infile, line)){
		count ++;
		istringstream iss(line);
		string wkt;

		getline(iss, wkt); // 提取wkt
		if (wkt.empty()) {
            std::cerr << "read WKT failed\n";
            continue;
        }
		// 去掉引号
        // if (wkt.size() > 0 && wkt.front() == '"') {
        //     wkt.erase(0, 1); // 删除起始的引号
        // }
        // if (wkt.size() > 0 && wkt.back() == '"') {
        //     wkt.erase(wkt.size() - 1); // 删除结尾的引号
        // }
		wkts.push_back(wkt);
	}

	infile.close();

	Point *points = new Point[count];

	for(int i = 0; i < count; i ++){
		size_t offset = wkts[i].find('(');
		points[i] = *read_vertices(wkts[i].c_str(), offset, false)->p;
	}

	return points;	
}

void dump_to_file(const char *path, char *data, size_t size)
{
	ofstream os;
	os.open(path, ios::out | ios::binary | ios::trunc);
	assert(os.is_open());

	os.write(data, size);
	os.close();
}

/*
 * in this file we define the .idl file format
 *
 * */

void dump_polygons_to_file(vector<Ideal *> ideals, const char *path){
	ofstream os;
	os.open(path, ios::out | ios::binary |ios::trunc);
	assert(os.is_open());

	size_t buffer_size = 100*1024*1024;
	char *data_buffer = new char[buffer_size];
	size_t data_size = 0;
	size_t curoffset = 0;
	PolygonMeta *pmeta = new PolygonMeta[ideals.size()];
	for(int i=0;i<ideals.size();i++){
		Ideal *p = ideals[i];
		if(p->get_data_size()+data_size > buffer_size){
			os.write(data_buffer, data_size);
			data_size = 0;
		}
		pmeta[i] = p->get_meta();
		pmeta[i].offset = curoffset;
		data_size += p->encode(data_buffer+data_size);
		curoffset += p->get_data_size();
	}

	// dump the rest polygon data
	if(data_size!=0){
		os.write(data_buffer, data_size);
	}
	// dump the meta data of the polygons
	os.write((char *)pmeta, sizeof(PolygonMeta)*ideals.size());
//	for(PolygonMeta &pm:pmeta){
//		printf("%ld\t%d\t%.12f\n", pm.offset, pm.size, pm.mbr.area());
//	}
	size_t bs = ideals.size();
	os.write((char *)&bs, sizeof(size_t));
	os.close();
	delete []pmeta;
}

MyPolygon *read_polygon_binary_file(ifstream &infile)
{
	size_t num_holes = 0;
	infile.read((char *)&num_holes, sizeof(size_t));

	size_t num_vertices = 0;
	infile.read((char *)&num_vertices, sizeof(size_t));
	if (num_vertices == 0)
	{
		return NULL;
	}
	MyPolygon *poly = new MyPolygon();
	VertexSequence *boundary = poly->get_boundary(num_vertices);
	infile.read((char *)boundary->p, num_vertices * sizeof(Point));
	if (boundary->clockwise())
	{
		boundary->reverse();
	}

	for (int i = 0; i < num_holes; i++)
	{
		infile.read((char *)&num_vertices, sizeof(long));
		assert(num_vertices);
		VertexSequence *vs = new VertexSequence(num_vertices);
		infile.read((char *)vs->p, num_vertices * sizeof(Point));
		if (!vs->clockwise())
		{
			vs->reverse();
		}
		poly->get_holes().push_back(vs);
	}
	return poly;
}

// idx starting from 0
MyPolygon *load_binary_file_single(const char *path, query_context ctx, int idx)
{
	ifstream infile;
	infile.open(path, ios::in | ios::binary);

	size_t num_polygons_infile;
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	assert(idx < num_polygons_infile && "the idx must smaller than the polygon number ");

	PolygonMeta pmeta;
	infile.seekg(-sizeof(size_t) - sizeof(PolygonMeta) * (num_polygons_infile - idx), infile.end);
	infile.read((char *)&pmeta, sizeof(PolygonMeta));

	char *buffer = new char[pmeta.size];

	infile.seekg(pmeta.offset, infile.beg);
	infile.read(buffer, pmeta.size);

	MyPolygon *poly = new MyPolygon();
	poly->decode(buffer);

	delete[] buffer;
	infile.close();
	return poly;
}

typedef struct
{
	ifstream *infile;
	size_t offset;
	size_t poly_size;
	size_t load(char *buffer)
	{
		infile->seekg(offset, infile->beg);
		infile->read(buffer, poly_size);
		return poly_size;
	}
} load_holder;

box universe_space(const char *path)
{
	if (!file_exist(path))
	{
		log("%s does not exist", path);
		exit(0);
	}
	struct timeval start = get_cur_time();

	ifstream infile;
	infile.open(path, ios::in | ios::binary);
	size_t num_polygons_infile = 0;
	infile.seekg(0, infile.end);
	// seek to the first polygon
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	assert(num_polygons_infile > 0 && "the file should contain at least one polygon");

	PolygonMeta *pmeta = new PolygonMeta[num_polygons_infile];
	infile.seekg(-sizeof(size_t) - sizeof(PolygonMeta) * num_polygons_infile, infile.end);
	infile.read((char *)pmeta, sizeof(PolygonMeta) * num_polygons_infile);

	box universe;
	for (size_t i = 0; i < num_polygons_infile; i++)
	{
		universe.update(pmeta[i].mbr);
	}

	return universe;
}

size_t number_of_objects(const char *path)
{
	if (!file_exist(path))
	{
		log("%s does not exist", path);
		exit(0);
	}
	ifstream infile;
	infile.open(path, ios::in | ios::binary);
	size_t num_polygons_infile = 0;
	infile.seekg(0, infile.end);
	// seek to the first polygon
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	return num_polygons_infile;
}

const size_t buffer_size = 10 * 1024 * 1024;

void *load_unit(void *arg)
{
	query_context *ctx = (query_context *)arg;
	vector<load_holder *> *jobs = (vector<load_holder *> *)ctx->target;
	vector<Ideal *> *global_polygons = (vector<Ideal *> *)ctx->target2;

	char *buffer = new char[buffer_size];
	vector<Ideal *> polygons;
	while (ctx->next_batch(1))
	{
		for (int i = ctx->index; i < ctx->index_end; i++)
		{
			load_holder *lh = (*jobs)[i];
			ctx->global_ctx->lock();
			size_t poly_size = lh->load(buffer);
			ctx->global_ctx->unlock();
			size_t off = 0;
			while (off < poly_size)
			{
				Ideal *poly = new Ideal();
				off += poly->decode(buffer + off);
				if (poly->get_num_vertices() >= 3 && tryluck(ctx->sample_rate))
				{
					polygons.push_back(poly);
					poly->getMBB();
				}
				else
				{
					delete poly;
				}
			}
		}
	}

	delete[] buffer;
	ctx->global_ctx->lock();
	global_polygons->insert(global_polygons->end(), polygons.begin(), polygons.end());
	ctx->global_ctx->unlock();
	polygons.clear();
	return NULL;
}
vector<Ideal *> load_binary_file(const char *path, query_context &global_ctx)
{
	global_ctx.index = 0;
	global_ctx.index_end = 0;
	vector<Ideal *> polygons;
	if (!file_exist(path))
	{
		log("%s does not exist", path);
		exit(0);
	}
	struct timeval start = get_cur_time();

	ifstream infile;
	infile.open(path, ios::in | ios::binary);
	size_t num_polygons_infile = 0;
	infile.seekg(0, infile.end);
	// seek to the first polygon
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	assert(num_polygons_infile > 0 && "the file should contain at least one polygon");

	PolygonMeta *pmeta = new PolygonMeta[num_polygons_infile];
	infile.seekg(-sizeof(size_t) - sizeof(PolygonMeta) * num_polygons_infile, infile.end);
	infile.read((char *)pmeta, sizeof(PolygonMeta) * num_polygons_infile);
	// the last one is the end
	size_t num_polygons = min(num_polygons_infile, global_ctx.max_num_polygons);

	logt("loading %ld polygon from %s", start, num_polygons, path);
	// organizing tasks
	vector<load_holder *> tasks;
	size_t cur = 0;
	while (cur < num_polygons)
	{
		size_t end = cur + 1;
		while (end < num_polygons &&
			   pmeta[end].offset - pmeta[cur].offset + pmeta[end].size < buffer_size)
		{
			end++;
		}
		load_holder *lh = new load_holder();
		lh->infile = &infile;
		lh->offset = pmeta[cur].offset;
		if (end < num_polygons)
		{
			lh->poly_size = pmeta[end].offset - pmeta[cur].offset;
		}
		else
		{
			lh->poly_size = pmeta[end - 1].offset - pmeta[cur].offset + pmeta[end - 1].size;
		}
		tasks.push_back(lh);
		cur = end;
	}

	logt("packed %ld tasks", start, tasks.size());

	global_ctx.target_num = tasks.size();
	pthread_t threads[global_ctx.num_threads];
	query_context myctx[global_ctx.num_threads];
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		myctx[i].index = 0;
		myctx[i] = global_ctx;
		myctx[i].thread_id = i;
		myctx[i].global_ctx = &global_ctx;
		myctx[i].target = (void *)&tasks;
		myctx[i].target2 = (void *)&polygons;
	}
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		pthread_create(&threads[i], NULL, load_unit, (void *)&myctx[i]);
	}

	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		void *status;
		pthread_join(threads[i], &status);
	}
	infile.close();
	delete[] pmeta;
	for (load_holder *lh : tasks)
	{
		delete lh;
	}
	logt("loaded %ld polygons", start, polygons.size());
	global_ctx.index = 0;
	for(int i = 0; i < polygons.size(); i ++) polygons[i]->id = i; 
	return polygons;
}

void *load_polygons_unit(void *arg)
{
	query_context *ctx = (query_context *)arg;
	vector<load_holder *> *jobs = (vector<load_holder *> *)ctx->target;
	vector<MyPolygon *> *global_polygons = (vector<MyPolygon *> *)ctx->target2;

	char *buffer = new char[buffer_size];
	vector<MyPolygon *> polygons;
	while (ctx->next_batch(1))
	{
		for (int i = ctx->index; i < ctx->index_end; i++)
		{
			load_holder *lh = (*jobs)[i];
			ctx->global_ctx->lock();
			size_t poly_size = lh->load(buffer);
			ctx->global_ctx->unlock();
			size_t off = 0;
			while (off < poly_size)
			{
				MyPolygon *poly = new MyPolygon();
				off += poly->decode(buffer + off);
				if (poly->get_num_vertices() >= 3 && tryluck(ctx->sample_rate))
				{
					polygons.push_back(poly);
					poly->getMBB();
				}
				else
				{
					delete poly;
				}
			}
		}
	}

	delete[] buffer;
	ctx->global_ctx->lock();
	global_polygons->insert(global_polygons->end(), polygons.begin(), polygons.end());
	ctx->global_ctx->unlock();
	polygons.clear();
	return NULL;
}

vector<MyPolygon *> load_polygons_from_path(const char *path, query_context &global_ctx)
{
	global_ctx.index = 0;
	global_ctx.index_end = 0;
	vector<MyPolygon *> polygons;
	if (!file_exist(path))
	{
		log("%s does not exist", path);
		exit(0);
	}
	struct timeval start = get_cur_time();

	ifstream infile;
	infile.open(path, ios::in | ios::binary);
	size_t num_polygons_infile = 0;
	infile.seekg(0, infile.end);
	// seek to the first polygon
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	assert(num_polygons_infile > 0 && "the file should contain at least one polygon");

	PolygonMeta *pmeta = new PolygonMeta[num_polygons_infile];
	infile.seekg(-sizeof(size_t) - sizeof(PolygonMeta) * num_polygons_infile, infile.end);
	infile.read((char *)pmeta, sizeof(PolygonMeta) * num_polygons_infile);
	// the last one is the end
	size_t num_polygons = min(num_polygons_infile, global_ctx.max_num_polygons);

	logt("loading %ld polygon from %s", start, num_polygons, path);
	// organizing tasks
	vector<load_holder *> tasks;
	size_t cur = 0;
	while (cur < num_polygons)
	{
		size_t end = cur + 1;
		while (end < num_polygons &&
			   pmeta[end].offset - pmeta[cur].offset + pmeta[end].size < buffer_size)
		{
			end++;
		}
		load_holder *lh = new load_holder();
		lh->infile = &infile;
		lh->offset = pmeta[cur].offset;
		if (end < num_polygons)
		{
			lh->poly_size = pmeta[end].offset - pmeta[cur].offset;
		}
		else
		{
			lh->poly_size = pmeta[end - 1].offset - pmeta[cur].offset + pmeta[end - 1].size;
		}
		tasks.push_back(lh);
		cur = end;
	}

	logt("packed %ld tasks", start, tasks.size());

	global_ctx.target_num = tasks.size();
	pthread_t threads[global_ctx.num_threads];
	query_context myctx[global_ctx.num_threads];
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		myctx[i].index = 0;
		myctx[i] = global_ctx;
		myctx[i].thread_id = i;
		myctx[i].global_ctx = &global_ctx;
		myctx[i].target = (void *)&tasks;
		myctx[i].target2 = (void *)&polygons;
	}
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		pthread_create(&threads[i], NULL, load_polygons_unit, (void *)&myctx[i]);
	}

	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		void *status;
		pthread_join(threads[i], &status);
	}
	infile.close();
	delete[] pmeta;
	for (load_holder *lh : tasks)
	{
		delete lh;
	}
	logt("loaded %ld polygons", start, polygons.size());
	return polygons;
}

size_t load_polygonmeta_from_file(const char *path, PolygonMeta **pmeta)
{
	ifstream infile;
	infile.open(path, ios::in | ios::binary);
	size_t num_polygons_infile = 0;
	infile.seekg(0, infile.end);
	// seek to the first polygon
	infile.seekg(-sizeof(size_t), infile.end);
	infile.read((char *)&num_polygons_infile, sizeof(size_t));
	assert(num_polygons_infile > 0 && "the file should contain at least one polygon");

	*pmeta = new PolygonMeta[num_polygons_infile];
	infile.seekg(-sizeof(size_t) - sizeof(PolygonMeta) * num_polygons_infile, infile.end);
	infile.read((char *)*pmeta, sizeof(PolygonMeta) * num_polygons_infile);

	return num_polygons_infile;
}

size_t load_mbr_from_file(const char *path, box **mbrs)
{

	if (!file_exist(path))
	{
		log("%s is empty", path);
		exit(0);
	}

	PolygonMeta *pmeta;
	size_t num_polygons = load_polygonmeta_from_file(path, &pmeta);
	*mbrs = new box[num_polygons];
	for (size_t i = 0; i < num_polygons; i++)
	{
		(*mbrs)[i] = pmeta[i].mbr;
	}
	delete[] pmeta;
	return num_polygons;
}

size_t load_points_from_path(const char *path, Point **points)
{
	size_t fsize = file_size(path);
	if (fsize <= 0)
	{
		log("%s is empty", path);
		exit(0);
	}
	size_t target_num = fsize / sizeof(Point);
	log_refresh("start loading %ld points", target_num);

	*points = new Point[target_num];
	ifstream infile(path, ios::in | ios::binary);
	infile.read((char *)*points, fsize);
	infile.close();
	return target_num;
}

VertexSequence *read_vertices(const char *wkt, size_t &offset, bool clockwise)
{
	// read until the left parenthesis
	skip_space(wkt, offset);
	assert(wkt[offset++] == '(');

	// count the number of vertices
	int cur_offset = offset;
	int num_vertices = 0;
	while (wkt[cur_offset++] != ')')
	{
		if (wkt[cur_offset] == ',')
		{
			num_vertices++;
		}
	}
	num_vertices++;
	VertexSequence *vs = new VertexSequence(num_vertices);

	// read x/y
	for (int i = 0; i < num_vertices; i++)
	{
		vs->p[i].x = read_float(wkt, offset);
		vs->p[i].y = read_float(wkt, offset);
	}
	if (clockwise)
	{
		if (!vs->clockwise())
		{
			vs->reverse();
		}
	}
	else
	{
		if (vs->clockwise())
		{
			vs->reverse();
		}
	}

	// read until the right parenthesis
	skip_space(wkt, offset);
	assert(wkt[offset++] == ')');
	return vs;
}

Ideal *read_polygon(const char *wkt, size_t &offset)
{

	Ideal *ideal = new Ideal();
	skip_space(wkt, offset);
	// left parentheses for the entire polygon
	assert(wkt[offset++] == '(');

	// read the vertices of the boundary polygon
	// the vertex must rotation in clockwise
	ideal->set_boundary(read_vertices(wkt, offset, false));
	if (ideal->get_boundary()->clockwise())
	{
		ideal->get_boundary()->reverse();
	}
	skip_space(wkt, offset);
	// polygons as the holes of the boundary polygon
	while (wkt[offset] == ',')
	{
		offset++;
		VertexSequence *vc = read_vertices(wkt, offset, true);
		if (!vc->clockwise())
		{
			vc->reverse();
		}
		ideal->get_holes().push_back(vc);

		skip_space(wkt, offset);
	}
	assert(wkt[offset++] == ')');
	ideal->getMBB();
	return ideal;
}
