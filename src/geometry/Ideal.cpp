#include "../include/Ideal.h"

Ideal::~Ideal()
{
	if (offset)
		delete[] offset;
	if (edge_sequences)
		delete[] edge_sequences;
	if (vertical)
		delete vertical;
	if (horizontal)
		delete horizontal;
}

void Ideal::add_edge(int idx, int start, int end)
{
	assert(end - start + 1 > 0);
	edge_sequences[idx] = make_pair(start, end - start + 1);
}

uint32_t Ideal::get_num_sequences(int id)
{
	if (show_status(id) != BORDER)
		return 0;
	return offset[id + 1] - offset[id];
}

void Ideal::init_edge_sequences(int num_edge_seqs)
{
	len_edge_sequences = num_edge_seqs;
	edge_sequences = new pair<uint32_t, uint32_t>[num_edge_seqs];
	assert(len_edge_sequences < 65536);
}

void Ideal::process_pixels_null(int x, int y)
{
	offset[x * y] = len_edge_sequences;
	for (int i = x * y - 1; i >= 0; i--)
	{
		if (show_status(i) != BORDER)
		{
			offset[i] = offset[i + 1];
		}
	}
}

void Ideal::process_crosses(map<int, vector<cross_info>> edges_info)
{
	int num_edge_seqs = 0;

	for (auto info : edges_info)
	{
		auto pix = info.first;
		auto crosses = info.second;

		if (crosses.size() == 0)
			return;

		if (crosses.size() % 2 == 1)
		{
			crosses.push_back(cross_info((cross_type)!crosses[crosses.size() - 1].type, crosses[crosses.size() - 1].edge_id));
		}

		int start = 0;
		int end = crosses.size() - 1;
		if (crosses[0].type == LEAVE)
		{
			assert(crosses[end].type == ENTER);
			num_edge_seqs += 2;
			start++;
			end--;
		}

		for (int i = start; i <= end; i++)
		{
			assert(crosses[i].type == ENTER);
			// special case, an ENTER has no pair LEAVE,
			// happens when one edge crosses the pair
			if (i == end || crosses[i + 1].type == ENTER)
			{
				num_edge_seqs++;
			}
			else
			{
				num_edge_seqs++;
				i++;
			}
		}
	}

	init_edge_sequences(num_edge_seqs);

	int idx = 0;
	int edge_count = 0;
	for (auto info : edges_info)
	{
		auto pix = info.first;
		auto crosses = info.second;

		if (crosses.size() == 0)
			return;

		if (crosses.size() % 2 == 1)
		{
			crosses.push_back(cross_info((cross_type)!crosses[crosses.size() - 1].type, crosses[crosses.size() - 1].edge_id));
		}

		assert(crosses.size() % 2 == 0);

		// Initialize based on crosses.size().
		int start = 0;
		int end = crosses.size() - 1;
		set_offset(pix, idx);

		if (crosses[0].type == LEAVE)
		{
			assert(crosses[end].type == ENTER);
			add_edge(idx++, 0, crosses[0].edge_id);
			add_edge(idx++, crosses[end].edge_id, boundary->num_vertices - 2);
			start++;
			end--;
		}

		for (int i = start; i <= end; i++)
		{
			assert(crosses[i].type == ENTER);
			// special case, an ENTER has no pair LEAVE,
			// happens when one edge crosses the pair
			if (i == end || crosses[i + 1].type == ENTER)
			{
				add_edge(idx++, crosses[i].edge_id, crosses[i].edge_id);
			}
			else
			{
				add_edge(idx++, crosses[i].edge_id, crosses[i + 1].edge_id);
				i++;
			}
		}
	}
}

void Ideal::process_intersection(map<int, vector<double>> intersection_info, Direction direction)
{
	int num_nodes = 0;
	for (auto i : intersection_info)
	{
		num_nodes += i.second.size();
	}
	if (direction == HORIZONTAL)
	{
		horizontal->init_intersection_node(num_nodes);
		horizontal->set_num_crosses(num_nodes);
		int idx = 0;
		for (auto info : intersection_info)
		{
			auto h = info.first;
			auto nodes = info.second;

			sort(nodes.begin(), nodes.end());

			horizontal->set_offset(h, idx);

			for (auto node : nodes)
			{
				horizontal->add_node(idx, node);
				idx++;
			}
		}
		horizontal->set_offset(dimy, idx);
	}
	else
	{
		vertical->init_intersection_node(num_nodes);
		vertical->set_num_crosses(num_nodes);
		vertical->set_offset(dimx + 1, num_nodes);

		int idx = 0;
		for (auto info : intersection_info)
		{
			auto h = info.first;
			auto nodes = info.second;

			sort(nodes.begin(), nodes.end());

			vertical->set_offset(h, idx);

			for (auto node : nodes)
			{
				vertical->add_node(idx, node);
				idx++;
			}
		}
		vertical->set_offset(dimx, idx);
	}
}

void Ideal::init_pixels()
{
	assert(mbr);

	status = new uint8_t[status_size];
	areas = new double[dimx * dimy]();
	memset(status, 0, status_size * sizeof(uint8_t));
	offset = new uint32_t[dimx * dimy + 1]; // +1 here is to ensure that pointer[num_pixels] equals len_edge_sequences, so we don't need to make a special case for the last pointer.
	horizontal = new Grid_line(dimy);
	vertical = new Grid_line(dimx);
}

void Ideal::evaluate_edges()
{
	map<int, vector<double>> horizontal_intersect_info;
	map<int, vector<double>> vertical_intersect_info;
	map<int, vector<cross_info>> edges_info;

	// normalize
	assert(mbr);
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	for (int i = 0; i < boundary->num_vertices - 1; i++)
	{
		double x1 = boundary->p[i].x;
		double y1 = boundary->p[i].y;
		double x2 = boundary->p[i + 1].x;
		double y2 = boundary->p[i + 1].y;

		int cur_startx = double_to_int((x1 - start_x) / step_x);
		int cur_endx = double_to_int((x2 - start_x) / step_x);
		int cur_starty = double_to_int((y1 - start_y) / step_y);
		int cur_endy = double_to_int((y2 - start_y) / step_y);

		if (cur_startx == dimx)
		{
			cur_startx--;
		}
		if (cur_endx == dimx)
		{
			cur_endx--;
		}

		int minx = min(cur_startx, cur_endx);
		int maxx = max(cur_startx, cur_endx);

		if (cur_starty == dimy)
		{
			cur_starty--;
		}
		if (cur_endy == dimy)
		{
			cur_endy--;
		}
		// todo should not happen for normal cases
		if (cur_startx >= dimx || cur_endx >= dimx || cur_starty >= dimy || cur_endy >= dimy)
		{
			cout << "xrange\t" << cur_startx << " " << cur_endx << endl;
			cout << "yrange\t" << cur_starty << " " << cur_endy << endl;
			printf("xrange_val\t%f %f\n", (x1 - start_x) / step_x, (x2 - start_x) / step_x);
			printf("yrange_val\t%f %f\n", (y1 - start_y) / step_y, (y2 - start_y) / step_y);
			assert(false);
		}
		assert(cur_startx < dimx);
		assert(cur_endx < dimx);
		assert(cur_starty < dimy);
		assert(cur_endy < dimy);

		set_status(get_id(cur_startx, cur_starty), BORDER);
		set_status(get_id(cur_endx, cur_endy), BORDER);

		// in the same pixel
		if (cur_startx == cur_endx && cur_starty == cur_endy)
		{
			continue;
		}

		if (y1 == y2)
		{
			// left to right
			if (cur_startx < cur_endx)
			{
				for (int x = cur_startx; x < cur_endx; x++)
				{
					vertical_intersect_info[x + 1].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x + 1, cur_starty)].push_back(cross_info(ENTER, i));
					set_status(get_id(x, cur_starty), BORDER);
					set_status(get_id(x + 1, cur_starty), BORDER);
				}
			}
			else
			{ // right to left
				for (int x = cur_startx; x > cur_endx; x--)
				{
					vertical_intersect_info[x].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x - 1, cur_starty)].push_back(cross_info(ENTER, i));
					set_status(get_id(x, cur_starty), BORDER);
					set_status(get_id(x - 1, cur_starty), BORDER);
				}
			}
		}
		else if (x1 == x2)
		{
			// bottom up
			if (cur_starty < cur_endy)
			{
				for (int y = cur_starty; y < cur_endy; y++)
				{
					horizontal_intersect_info[y + 1].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y + 1)].push_back(cross_info(ENTER, i));
					set_status(get_id(cur_startx, y), BORDER);
					set_status(get_id(cur_startx, y + 1), BORDER);
				}
			}
			else
			{ // border[bottom] down
				for (int y = cur_starty; y > cur_endy; y--)
				{
					horizontal_intersect_info[y].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y - 1)].push_back(cross_info(ENTER, i));
					set_status(get_id(cur_startx, y), BORDER);
					set_status(get_id(cur_startx, y - 1), BORDER);
				}
			}
		}
		else
		{
			// solve the line function
			double a = (y1 - y2) / (x1 - x2);
			double b = (x1 * y2 - x2 * y1) / (x1 - x2);

			int x = cur_startx;
			int y = cur_starty;
			while (x != cur_endx || y != cur_endy)
			{
				bool passed = false;
				double yval = 0;
				double xval = 0;
				int cur_x = 0;
				int cur_y = 0;
				// check horizontally
				if (x != cur_endx)
				{
					if (cur_startx < cur_endx)
					{
						xval = ((double)x + 1) * step_x + start_x;
					}
					else
					{
						xval = (double)x * step_x + start_x;
					}
					yval = xval * a + b;
					cur_y = (yval - start_y) / step_y;
					// printf("y %f %d\n",(yval-start_y)/step_y,cur_y);
					if (cur_y > max(cur_endy, cur_starty))
					{
						cur_y = max(cur_endy, cur_starty);
					}
					if (cur_y < min(cur_endy, cur_starty))
					{
						cur_y = min(cur_endy, cur_starty);
					}
					if (cur_y == y)
					{
						passed = true;
						// left to right
						if (cur_startx < cur_endx)
						{
							vertical_intersect_info[x + 1].push_back(yval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x++, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
						else
						{ // right to left
							vertical_intersect_info[x].push_back(yval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x--, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
					}
				}
				// check vertically
				if (y != cur_endy)
				{
					if (cur_starty < cur_endy)
					{
						yval = (y + 1) * step_y + start_y;
					}
					else
					{
						yval = y * step_y + start_y;
					}
					xval = (yval - b) / a;
					int cur_x = (xval - start_x) / step_x;
					// printf("x %f %d\n",(xval-start_x)/step_x,cur_x);
					if (cur_x > max(cur_endx, cur_startx))
					{
						cur_x = max(cur_endx, cur_startx);
					}
					if (cur_x < min(cur_endx, cur_startx))
					{
						cur_x = min(cur_endx, cur_startx);
					}
					if (cur_x == x)
					{
						passed = true;
						if (cur_starty < cur_endy)
						{ // bottom up
							horizontal_intersect_info[y + 1].push_back(xval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y++)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
						else
						{ // top down
							horizontal_intersect_info[y].push_back(xval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y--)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
					}
				}
				// for debugging, should never happen
				if (!passed)
				{
					boundary->print();
					cout << "dim\t" << dimx << " " << dimy << endl;
					printf("val\t%f %f\n", (xval - start_x) / step_x, (yval - start_y) / step_y);
					cout << "curxy\t" << x << " " << y << endl;
					cout << "calxy\t" << cur_x << " " << cur_y << endl;
					cout << "xrange\t" << cur_startx << " " << cur_endx << endl;
					cout << "yrange\t" << cur_starty << " " << cur_endy << endl;
					printf("xrange_val\t%f %f\n", (x1 - start_x) / step_x, (x2 - start_x) / step_x);
					printf("yrange_val\t%f %f\n", (y1 - start_y) / step_y, (y2 - start_y) / step_y);
				}
				assert(passed);
			}
		}
	}

	// special case
	if (edges_info.size() == 0 && boundary->num_vertices > 0)
	{
		init_edge_sequences(1);
		set_offset(0, 0);
		add_edge(0, 0, boundary->num_vertices - 1);
	}
	else
	{
		process_crosses(edges_info);
	}

	process_intersection(horizontal_intersect_info, HORIZONTAL);
	process_intersection(vertical_intersect_info, VERTICAL);
	process_pixels_null(dimx, dimy);
}

void Ideal::scanline_reandering()
{
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	for (int y = 1; y < dimy; y++)
	{
		bool isin = false;
		uint32_t i = horizontal->get_offset(y), j = horizontal->get_offset(y + 1);
		for (int x = 0; x < dimx; x++)
		{
			if (show_status(get_id(x, y)) != BORDER)
			{
				if (isin)
				{
					set_status(get_id(x, y), IN);
				}
				else
				{
					set_status(get_id(x, y), OUT);
				}
				continue;
			}
			int pass = 0;
			while (i < j && horizontal->get_intersection_nodes(i) <= start_x + step_x * (x + 1))
			{
				pass++;
				i++;
			}
			if (pass % 2 == 1)
				isin = !isin;
		}
	}
}

TempPolygon intersectionDualX(Ideal *pol, double &Xi, double &Xi1)
{
	double slope;
	double yi, yi1;
	TempPolygon clippedPolygon;

	// for each edge of the polygon
	//  ITERATES THE VERTICES WHICH MUST BE IN A COUNTER-CLOCKWISE DIRECTION!!!
	//  so that right of the line means inside, left means outside
	for (auto itV = 0; itV < pol->get_num_vertices() - 1; itV++)
	{
		Point pointA = pol->get_boundary()->p[itV];
		Point pointB = pol->get_boundary()->p[itV + 1];

		// check cases
		if (pointA.x == pointB.x)
		{
			// edge is vertical
			// set the lowest point as the intersection point
			yi = min(pointA.y, pointB.y);
			yi1 = min(pointA.y, pointB.y);
		}
		else
		{
			// solve for y
			slope = getSlope(pointA, pointB);
			// check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
			// find intersection points for both Xi and Xi1
			yi = slope * (Xi - pointB.x) + pointB.y;
			yi1 = slope * (Xi1 - pointB.x) + pointB.y;
		}

		// intersection points
		Point pi(Xi, yi);
		Point pi1(Xi1, yi1);

		// cout << "X CLIPPING " << endl;
		// POINT A
		if (isInsideVerticalDual(Xi, Xi1, pointA.x))
		{
			// printf("PointA ");
			// pointA.print();
			// printf("PointB ");
			// pointB.print();
			// printf("Intersection ");
			// pi.print();
			// pointA inside, clipping result
			clippedPolygon.addPoint(pointA);
			// cout << "point(" << pointA.x << " " << pointA.y << ") added" << endl;
		}

		// intersection points in PROPER ORDER
		if (pointA.x < pointB.x)
		{
			// first pi, then pi1
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
			{
				// intersection point pi is a clipping result
				clippedPolygon.addPoint(pi);
				// cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
			}

			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
			{
				// intersection point pi1 is a clipping result
				clippedPolygon.addPoint(pi1);
				// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
			}
		}
		else
		{
			// first pi1 then pi
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
			{
				// intersection point pi1 is a clipping result
				clippedPolygon.addPoint(pi1);
				// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
			}
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
			{
				// intersection point pi is a clipping result
				clippedPolygon.addPoint(pi);
				// cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
				// printf("point pi (%.12lf %.12lf) added\n", pi.x, pi.y);
			}
		}

		// POINT B
		if (isInsideVerticalDual(Xi, Xi1, pointB.x))
		{
			// pointB inside, clipping result
			clippedPolygon.addPoint(pointB);
			// cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
		}
	}
	// add the first point to the end
	//  so that the polygon "closes"
	if (clippedPolygon.vertices.size() != 0)
	{
		clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
	}

	return clippedPolygon;
}

vector<Point> intersectionDualY(TempPolygon &pol, double Yi, double Yi1)
{
	double slope;
	double xi, xi1;
	TempPolygon clippedPolygon;

	for (auto itV = pol.vertices.begin(); itV != pol.vertices.end() - 1; itV++)
	{
		Point pointA = *itV;
		Point pointB = *(itV + 1);

		// cout << "POINTA" << endl;
		// pointA.print();
		// pointB.print();
		// cout << "POINTB" << endl;

		bool parallels = false;
		// calculate the slope (if any)
		if (pointA.x == pointB.x)
		{
			// edge is vertical
			// the only possible point of intersection is y (Yi or Yi1)
			//  do nothing
			xi = pointA.x;
			xi1 = pointA.x;
		}
		else
		{
			slope = getSlope(pointA, pointB);
			// check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
			// this flag will skip unnecessary checks in this event
			if (pointA.y == pointB.y)
			{
				if (pointA.y == Yi)
				{
					xi = pointA.x;
					xi1 = -numeric_limits<double>::max();
				}
				else if (pointA.y == Yi1)
				{
					xi = -numeric_limits<double>::max();
					xi1 = pointA.x;
				}
				else
				{
					// THEY DO NOT INETERSECT, THEY ARE PARALLELS
					parallels = true;
				}
			}
			else
			{
				// solve for x
				xi = ((Yi - pointB.y) / slope) + pointB.x;
				xi1 = ((Yi1 - pointB.y) / slope) + pointB.x;
				// cout << "   solved for x" << endl;
			}
		}

		// POINT A
		if (isInsideHorizontalDual(Yi, Yi1, pointA.y))
		{
			// pointA inside, clipping result
			// clippedPolygon.addPoint(pointA);
			clippedPolygon.vertices.push_back(pointA);
			// cout << "point A (" << pointA.x << " " << pointA.y << ") added" << endl;
		}

		if (!parallels)
		{
			// intersection points
			Point pi(xi, Yi);
			Point pi1(xi1, Yi1);

			// intersection points in PROPER ORDER
			if (pointA.y < pointB.y)
			{
				// first pi, then pi1
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
				{
					// intersection point pi is a clipping result
					// clippedPolygon.addPoint(pi);
					clippedPolygon.vertices.push_back(pi);
					// cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
				}

				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
				{
					// intersection point pi1 is a clipping result
					// clippedPolygon.addPoint(pi1);
					clippedPolygon.vertices.push_back(pi1);
					// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
				}
			}
			else
			{
				// first pi1 then pi
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
				{
					// intersection point pi1 is a clipping result
					// clippedPolygon.addPoint(pi1);
					clippedPolygon.vertices.push_back(pi1);
					// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
				}
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
				{
					// intersection point pi is a clipping result
					// clippedPolygon.addPoint(pi);
					clippedPolygon.vertices.push_back(pi);
					// cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
				}
			}

			// POINT B
			// if (isInsideHorizontalDual(Yi, Yi1, pointB.y))
			// {
			// 	// pointB inside, clipping result
			// 	clippedPolygon.addPoint(pointB);
			// 	// cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
			// }
		}
	}
	// cout << "sort before" << endl;
	// for (auto x : clippedPolygon.vertices)
	//     x.print();
	// sort points
	// sort_by_polar_angle(clippedPolygon.vertices);

	// cout << "sort after" << endl;
	// for (auto x : clippedPolygon.vertices)
	//     x.print();

	//"close" the polygon (first and last points in order must be the same point)
	if (clippedPolygon.vertices.size() != 0 && clippedPolygon.vertices.front() != clippedPolygon.vertices.back())
	{
		clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
	}

	return clippedPolygon.vertices;
}

void Ideal::calculate_fullness()
{
	double Xi, Xi1, Yi, Yi1;
	double kx, ky;

	vector<Point> clippedPoints;
	vector<TempPolygon> subpolygonsAfterX;

	Xi = getMBB()->low[0];
	Xi1 = Xi + step_x;

	kx = Xi + dimx * step_x;

	TempPolygon tempPol;
	subpolygonsAfterX.reserve(dimx);

	int x = 0;
	while (Xi1 < kx + 1e-9)
	{
		tempPol = intersectionDualX(this, Xi, Xi1);

		// if(id == 0){
		// 	printf("----------------------------------------------------\n");
		// 	printf("tempPol%d: %lf %lf\n", x, tempPol.cellX, tempPol.cellY);
		// 	for (auto x : tempPol.vertices)
		// 		x.print();
		// 	printf("----------------------------------------------------\n");
		// }

		if (tempPol.vertices.size() > 0)
		{
			tempPol.cellX = x;
			subpolygonsAfterX.push_back(tempPol);
		}

		// move both vertical lines equally
		Xi += step_x;
		Xi1 = Xi + step_x;
		x++;
	}

	ky = getMBB()->low[1] + dimy * step_y;

	int type;

	auto it = subpolygonsAfterX.begin();
	while (it != subpolygonsAfterX.end())
	{
		// FOR NORMALIZED
		Yi = getMBB()->low[1];
		Yi1 = Yi + step_y;

		int y = 0;
		// sweep the y axis getting pairs of horizontal lines Yi & Yi+1
		while (Yi1 < ky + 1e-9)
		{
			// returns the subpolygon furtherly clipped in the y axis by Yi and Yi+1
			clippedPoints = intersectionDualY(*it, Yi, Yi1);

			// this helps ignore a large portion of the empty cells for a polygon
			// if (clippedPoints.size() > 2)
			// {
			// calculate its area and classify it

			double clippedArea = computePolygonArea(clippedPoints);
			type = classifySubpolygon(clippedArea, step_x * step_y, category_count);

			// if(id == 0){
			// 	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");
			// 	for (auto point : clippedPoints)
			// 		point.print();
			// 	printf("x = %d y = %d area = %.16lf pixelArea = %.16lf type = %d\n", it->cellX, y, clippedArea, step_x * step_y, type);

			// 	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");
			// }
			int pix_id = get_id(it->cellX, y);
			assert(pix_id < dimx * dimy);
			if (status[pix_id] == BORDER)
			{
				status[pix_id] = max(1, min(type, category_count - 2));
				areas[pix_id] = clippedArea;
			}
			else if (status[pix_id] == IN)
			{
				status[pix_id] = category_count - 1;
				areas[pix_id] = get_pixel_area();
			}else{
				status[pix_id] = 0;
				areas[pix_id] = 0.0;
			}

			// move the horizontal lines equally to the next position
			Yi += step_y;
			Yi1 = Yi + step_y;
			y++;
		}
		it++;
	}
}

void Ideal::rasterization()
{

	// 1. create space for the pixels
	init_pixels();

	// 3. edge crossing to identify BORDER pixels
	evaluate_edges();

	// 3. determine the status of rest pixels with scanline rendering
	scanline_reandering();

	// 2. determine the fullness of pixels
	calculate_fullness();
}

void Ideal::rasterization(int vpr)
{
	assert(vpr > 0);
	pthread_mutex_lock(&ideal_partition_lock);

	rasterization();

	if (use_hierachy)
	{
		layers[num_layers].set_status(status);

		for (int i = num_layers - 1; i >= 0; i--)
		{
			merge_status(layers[i]);
			memcpy(status + layer_offset[i], layers[i].get_status(), layers[i].get_dimx() * layers[i].get_dimy() * sizeof(uint8_t));
		}
	}

	pthread_mutex_unlock(&ideal_partition_lock);
}

int Ideal::num_edges_covered(int id)
{
	int c = 0;
	for (int i = 0; i < get_num_sequences(id); i++)
	{
		auto r = edge_sequences[offset[id] + i];
		c += r.second;
	}
	return c;
}

int Ideal::get_num_border_edge()
{
	int num = 0;
	for (int i = 0; i < get_num_pixels(); i++)
	{
		if (show_status(i) == BORDER)
		{
			num += num_edges_covered(i);
		}
	}
	return num;
}

size_t Ideal::get_num_crosses()
{
	size_t num = 0;
	num = horizontal->get_num_crosses() + vertical->get_num_crosses();
	return num;
}

int Ideal::count_intersection_nodes(Point &p)
{
	// here we assume the point inside one of the pixel
	int pix_id = get_pixel_id(p);
	assert(show_status(pix_id) == BORDER);
	int count = 0;
	int x = get_x(pix_id) + 1;
	uint32_t i = vertical->get_offset(x), j = vertical->get_offset(x + 1);
	// if (x < dimx)
	// 	j = vertical->get_offset(x + 1);
	// else
	//	j = vertical->get_num_crosses();
	while (i < j && vertical->get_intersection_nodes(i) < p.y)
	{
		count++;
		i++;
	}
	return count;
}

double Ideal::merge_area(box target, PartitionStatus &st)
{
	int start_x = get_offset_x(target.low[0]);
	int start_y = get_offset_y(target.low[1]);
	int end_x = get_offset_x(target.high[0]) - 1;
	int end_y = get_offset_y(target.high[1]) - 1;

	assert(start_x >= 0 && start_y >= 0 && end_x < dimx && end_y < dimy);

	if (end_y < start_y)
		end_y = start_y;
	if (end_x < start_x)
		end_x = start_x;

	double clippedArea = 0.0;

	for (int i = start_x; i <= end_x; i++)
	{
		for (int j = start_y; j <= end_y; j++)
		{
			
			int id = get_id(i, j);
			if(show_status(id) == BORDER) st = BORDER;
			clippedArea += areas[id];
		}
	}
	if(st != BORDER) st = show_status(get_id(start_x, start_y));
	// printf("clippedArea = %.12lf\n", clippedArea);

	return clippedArea;
}

void Ideal::merge_status(Hraster &r)
{
	for (int x = 0; x < r.get_dimx(); x++)
	{
		for (int y = 0; y < r.get_dimy(); y++)
		{
			PartitionStatus st;
			double mergedArea = merge_area(r.get_pixel_box(x, y), st);
			uint8_t fullness;
			if(st == BORDER){
				fullness = classifySubpolygon(mergedArea, r.get_step_x() * r.get_step_y(), category_count);
			}else if(st == IN){
				fullness = category_count - 1;
			}else{
				fullness = 0;
			}
			r.set_status(r.get_id(x, y), fullness);
		}
	}
}

void Ideal::layering()
{
	num_layers = static_cast<int>(ceil(max(log(dimx) / log(2.0), log(dimy) / log(2.0))));
	layers = new Hraster[num_layers + 1];
	layer_offset = new uint32_t[num_layers + 1];
	layer_info = new RasterInfo[num_layers + 1];

	// process the last layer
	layers[num_layers].init(step_x, step_y, dimx, dimy, getMBB(), true);
	layer_info[num_layers] = {*mbr, dimx, dimy, step_x, step_y};
	layer_offset[num_layers] = 0;

	status_size = get_num_pixels();

	for (int i = num_layers - 1; i >= 0; i--)
	{
		double _step_x = layers[i + 1].get_step_x() * 2, _step_y = layers[i + 1].get_step_y() * 2;
		int _dimx = static_cast<int>(ceil(layers[i + 1].get_dimx() / 2.0)), _dimy = static_cast<int>(ceil(layers[i + 1].get_dimy() / 2.0));

		layers[i].init(_step_x, _step_y, _dimx, _dimy, getMBB(), false);
		layer_info[i] = {*layers[i].mbr, layers[i].get_dimx(), layers[i].get_dimy(), _step_x, _step_y};
		layer_offset[i] = status_size;
		status_size += layers[i].get_dimx() * layers[i].get_dimy();
	}
}

bool Ideal::contain(Point &p, query_context *ctx, bool profile)
{

	// the MBB may not be checked for within query
	if (!mbr->contain(p))
	{
		return false;
	}

	struct timeval start = get_cur_time();
	// todo adjust the lower bound of pixel number when the raster model is usable
	start = get_cur_time();
	int target = get_pixel_id(p);
	box bx = get_pixel_box(get_x(target), get_y(target));
	double bx_high = bx.high[0];
	if (show_status(target) == IN)
	{
		return true;
	}
	if (show_status(target) == OUT)
	{
		return false;
	}

	start = get_cur_time();
	bool ret = false;

	// checking the intersection edges in the target pixel
	for (uint32_t e = 0; e < get_num_sequences(target); e++)
	{
		auto edges = get_edge_sequence(get_offset(target) + e);
		auto pos = edges.first;
		for (int k = 0; k < edges.second; k++)
		{
			int i = pos + k;
			int j = i + 1; // ATTENTION
			if (((boundary->p[i].y >= p.y) != (boundary->p[j].y >= p.y)))
			{
				double int_x = (boundary->p[j].x - boundary->p[i].x) * (p.y - boundary->p[i].y) / (boundary->p[j].y - boundary->p[i].y) + boundary->p[i].x;
				if (p.x <= int_x && int_x <= bx_high)
				{
					ret = !ret;
				}
			}
		}
	}
	// check the crossing nodes on the right bar
	// swap the state of ret if odd number of intersection
	// nodes encountered at the right side of the border
	struct timeval tstart = get_cur_time();
	int nc = count_intersection_nodes(p);
	if (nc % 2 == 1)
	{
		ret = !ret;
	}
	return ret;
}

PartitionStatus Ideal::segment_contain(Point &p)
{
	int target = get_pixel_id(p);

	box bx = get_pixel_box(get_x(target), get_y(target));
	double bx_high = bx.high[0];
	if (show_status(target) == IN)
	{
		return IN;
	}
	if (show_status(target) == OUT)
	{
		return OUT;
	}

	bool ret = false;

	// checking the intersection edges in the target pixel
	for (uint32_t e = 0; e < get_num_sequences(target); e++)
	{
		auto edges = get_edge_sequence(get_offset(target) + e);
		auto pos = edges.first;
		for (int k = 0; k < edges.second; k++)
		{
			Point v1 = boundary->p[pos + k];
			Point v2 = boundary->p[pos + k + 1];
			// if (abs(p.x - 133.967605) < 1e-9 && abs(p.y - 34.558846) < 1e-9)
			// {
			// 	printf("----------------------CHECK-----------------------------\n");
			// 	p.print();
			// 	v1.print();
			// 	v2.print();
			// 	printf("----------------------CHECK-----------------------------\n");
			// }

			if (p == v1 || p == v2)
			{
				// printf("OUTPUT1\n");
				// p.print();
				// v1.print();
				// v2.print();
				return BORDER;
			}

			if ((v1.y >= p.y) != (v2.y >= p.y))
			{

				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (abs(dy) > 1e-9)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if (fabs(p.x - int_x) < 1e-9)
					{
						// printf("OUTPUT2\n");
						// p.print();
						// v1.print();
						// v2.print();
						return BORDER;
					}
					if (p.x < int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
			else if (v1.y == p.y && v2.y == p.y && (v1.x >= p.x) != (v2.x >= p.x))
			{
				// printf("OUTPUT3\n");
				// p.print();
				// v1.print();
				// v2.print();
				return BORDER;
			}
		}
	}
	// check the crossing nodes on the right bar
	// swap the state of ret if odd number of intersection
	// nodes encountered at the right side of the border
	int nc = count_intersection_nodes(p);
	if (nc % 2 == 1)
	{
		ret = !ret;
	}
	if (ret)
	{
		return IN;
	}
	else
	{
		return OUT;
	}
}

bool Ideal::contain(Ideal *target, query_context *ctx, bool profile)
{
	if (!getMBB()->contain(*target->getMBB()))
	{
		// log("mbb do not contain");
		return false;
	}
	vector<int> pxs = retrieve_pixels(target->getMBB());
	int etn = 0;
	int itn = 0;
	for (auto p : pxs)
	{
		if (show_status(p) == OUT)
		{
			etn++;
		}
		else if (show_status(p) == IN)
		{
			itn++;
		}
	}
	if (etn == pxs.size())
	{
		return false;
	}
	if (itn == pxs.size())
	{
		return true;
	}

	vector<int> tpxs;

	for (auto p : pxs)
	{
		box bx = get_pixel_box(get_x(p), get_y(p));
		tpxs = target->retrieve_pixels(&bx);
		for (auto p2 : tpxs)
		{
			// an external pixel of the container intersects an internal
			// pixel of the containee, which means the containment must be false
			if (show_status(p) == IN)
				continue;
			if (show_status(p) == OUT && target->show_status(p2) == IN)
			{
				return false;
			}
			if (show_status(p) == OUT && target->show_status(p2) == BORDER)
			{
				Point pix_border[5];
				pix_border[0].x = bx.low[0];
				pix_border[0].y = bx.low[1];
				pix_border[1].x = bx.low[0];
				pix_border[1].y = bx.high[1];
				pix_border[2].x = bx.high[0];
				pix_border[2].y = bx.high[1];
				pix_border[3].x = bx.high[0];
				pix_border[3].y = bx.low[1];
				pix_border[4].x = bx.low[0];
				pix_border[4].y = bx.low[1];
				for (int e = 0; e < target->get_num_sequences(p2); e++)
				{
					auto edges = target->get_edge_sequence(target->get_offset(p2) + e);
					auto pos = edges.first;
					auto size = edges.second;
					if (segment_intersect_batch(target->boundary->p + pos, pix_border, size, 4))
					{
						return false;
					}
				}
			}
			// evaluate the state
			if (show_status(p) == BORDER && target->show_status(p2) == BORDER)
			{
				for (int i = 0; i < get_num_sequences(p); i++)
				{
					auto r = get_edge_sequence(get_offset(p) + i);
					for (int j = 0; j < target->get_num_sequences(p2); j++)
					{
						auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
						if (segment_intersect_batch(boundary->p + r.first, target->boundary->p + r2.first, r.second, r2.second))
						{
							return false;
						}
					}
				}
			}
		}
		tpxs.clear();
	}
	pxs.clear();

	// this is the last step for all the cases, when no intersection segment is identified
	// pick one point from the target and it must be contained by this polygon
	Point p(target->getx(0), target->gety(0));
	return contain(p, ctx, false);
}

inline int binary_search(vector<Segment> &sorted_array, int left, int right, Point target)
{
	while (left < right)
	{
		int mid = (left + right) >> 1;
		if (target <= sorted_array[mid].start)
			right = mid;
		else
			left = mid + 1;
	}
	if (sorted_array[left].start == target)
		return left;
	else
		return -1; // Not Found
}

void Ideal::intersection(Ideal *target, query_context *ctx)
{
	vector<int> pxs = retrieve_pixels(target->getMBB());
	int etn = 0;
	int itn = 0;
	for (auto p : pxs)
	{
		if (show_status(p) == OUT)
		{
			etn++;
		}
		else if (show_status(p) == IN)
		{
			itn++;
		}
	}
	if (etn == pxs.size() || itn == pxs.size())
	{
		return;
	}

	vector<int> tpxs;
	vector<Intersection> inters;

	for (auto p : pxs)
	{
		box bx = get_pixel_box(get_x(p), get_y(p));
		tpxs = target->retrieve_pixels(&bx);
		for (auto p2 : tpxs)
		{
			if (show_status(p) == BORDER && target->show_status(p2) == BORDER)
			{
				for (int i = 0; i < get_num_sequences(p); i++)
				{
					auto r = get_edge_sequence(get_offset(p) + i);
					for (int j = 0; j < target->get_num_sequences(p2); j++)
					{
						auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
						assert(r.second != 0 && r2.second != 0);
						segment_intersect_batch(boundary->p, target->boundary->p, r.first, r2.first, r.first + r.second, r2.first + r2.second, inters);
					}
				}
			}
		}
		tpxs.clear();
	}
	pxs.clear();

	std::sort(inters.begin(), inters.end(),
			  [](const Intersection &a, const Intersection &b)
			  {
				  if (a.p.x != b.p.x)
				  {
					  return a.p.x < b.p.x;
				  }
				  return a.p.y < b.p.y;
			  });

	auto new_end = std::unique(inters.begin(), inters.end(),
							   [](const Intersection &a, const Intersection &b)
							   {
								   return a.p == b.p;
							   });

	inters.erase(new_end, inters.end());

	int num_inters = inters.size();

	// printf("num_inters = %d\n", num_inters);

	// for (auto inter : inters)
	// {
	// 	inter.print();
	// }

	// return;

	std::sort(inters.begin(), inters.end(),
			  [](const Intersection &a, const Intersection &b)
			  {
				  if (a.edge_source_id != b.edge_source_id)
				  {
					  return a.edge_source_id < b.edge_source_id;
				  }
				  return a.t < b.t;
			  });

	vector<Segment> segments;

	for (int i = 0; i < num_inters; i++)
	{
		Intersection a = inters[i];
		Intersection b = inters[(i + 1) % num_inters];

		if (a.p == b.p)
			continue;

		int a_edge_id = a.edge_source_id;
		int b_edge_id = b.edge_source_id;
		double a_param = a.t;
		double b_param = b.t;

		if (fabs(a_param - 1.0) < eps)
		{
			a_edge_id = (a_edge_id + 1) % (get_num_vertices() - 1);
			a_param = 0.0;
		}
		if (fabs(b_param) < eps)
		{
			b_edge_id--;
			b_param = 1.0;
		}

		segments.push_back({true, a.p, b.p,
							(a_edge_id == b_edge_id && a_param < b_param) ? -1 : a_edge_id + 1,
							(a_edge_id == b_edge_id && a_param < b_param) ? -1 : b_edge_id,
							0});
	}

	std::sort(inters.begin(), inters.end(),
			  [](const Intersection &a, const Intersection &b)
			  {
				  if (a.edge_target_id != b.edge_target_id)
				  {
					  return a.edge_target_id < b.edge_target_id;
				  }
				  return a.u < b.u;
			  });

	for (int i = 0; i < num_inters; i++)
	{
		Intersection a = inters[i];
		Intersection b = inters[(i + 1) % num_inters];

		if (a.p == b.p)
			continue;

		int a_edge_id = a.edge_target_id;
		int b_edge_id = b.edge_target_id;
		double a_param = a.u;
		double b_param = b.u;

		if (fabs(a_param - 1.0) < eps)
		{
			a_edge_id = (a_edge_id + 1) % (target->get_num_vertices() - 1);
			a_param = 0.0;
		}
		if (fabs(b_param) < eps)
		{
			b_edge_id--;
			b_param = 1.0;
		}

		segments.push_back({false, a.p, b.p,
							(a_edge_id == b_edge_id && a_param < b_param) ? -1 : a_edge_id + 1,
							(a_edge_id == b_edge_id && a_param < b_param) ? -1 : b_edge_id,
							0});

		// if (a_edge_id == b_edge_id)
		// {
		// 	if (a_param < b_param)
		// 		segments.push_back({false, a.p, b.p, -1, -1, 0});
		// 	else
		// 		segments.push_back({false, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
		// }
		// else
		// {
		// 	segments.push_back({false, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
		// }
	}

	auto num_segments = segments.size();
	// printf("num_segments = %d\n", num_segments);

	sort(segments.begin(), segments.end(),
		 [](const Segment &a, const Segment &b)
		 {
			 if (fabs(a.start.x - b.start.x) >= 1e-9)
			 {
				 return a.start.x < b.start.x;
			 }
			 else if (fabs(a.start.y - b.start.y) >= 1e-9)
			 {
				 return a.start.y < b.start.y;
			 }
			 else if (fabs(a.end.x - b.end.x) >= 1e-9)
			 {
				 return a.end.x < b.end.x;
			 }
			 else if (fabs(a.end.y - b.end.y) >= 1e-9)
			 {
				 return a.end.y < b.end.y;
			 }
			 else
			 {
				 return a.is_source < b.is_source;
			 }
		 });

	vector<PartitionStatus> status;
	for (auto seg : segments)
	{
		Point p;
		if (seg.edge_start == -1)
		{
			p = (seg.start + seg.end) * 0.5;
		}
		else
		{
			p = seg.is_source ? boundary->p[seg.edge_start] : target->boundary->p[seg.edge_start];
		}
		if (seg.is_source)
			status.push_back(target->segment_contain(p));
		else
			status.push_back(segment_contain(p));
	}

	// for (int i = 0; i < num_segments; i++)
	// {
	// 	segments[i].print();
	// 	printf("%d\n", status[i]);
	// }

	// return;

	for (size_t startIdx = 0; startIdx < segments.size(); startIdx++)
	{
		if (startIdx < segments.size() - 1 && (status[startIdx] == OUT || status[startIdx] == BORDER))
			continue;
			
		if (startIdx == segments.size() - 1 && status[startIdx] == OUT)
			continue;

		if (startIdx == segments.size() - 1 && status[startIdx] == BORDER && ctx->intersection_polygons.size() > 0){
			continue;
		}

		size_t currentSegIdx = startIdx;
		Point currentPoint = segments[startIdx].start;
		Point startPoint = currentPoint;
		// printf("START POINT(%lf %lf)\n", startPoint.x, startPoint.y);
		vector<Point> currentVertices;

		bool foundCycle = false;

		while (status[currentSegIdx])
		{
			status[currentSegIdx] = OUT;
			// printf("SWITCH POINT(%lf %lf)\n", currentPoint.x, currentPoint.y);
			currentVertices.push_back(currentPoint);
			const Segment &seg = segments[currentSegIdx];
			Point *vertices = seg.is_source ? boundary->p : target->boundary->p;
			size_t num_vertices = seg.is_source ? get_num_vertices() : target->get_num_vertices();

			if (seg.edge_start != -1)
			{
				if (seg.edge_start <= seg.edge_end)
				{
					for (int verId = seg.edge_start; verId <= seg.edge_end; verId++)
					{
						// printf("POINT (%lf %lf)\n", vertices[verId].x, vertices[verId].y);
						currentVertices.push_back(vertices[verId]);
					}
				}
				else
				{
					for (int verId = seg.edge_start; verId < num_vertices - 1; verId++)
					{
						// printf("POINT (%lf %lf)\n", vertices[verId].x, vertices[verId].y);
						currentVertices.push_back(vertices[verId]);
					}
					for (int verId = 0; verId <= seg.edge_end; verId++)
					{
						// printf("POINT (%lf %lf)\n", vertices[verId].x, vertices[verId].y);
						currentVertices.push_back(vertices[verId]);
					}
				}
			}

			Point nextPoint = currentPoint == seg.start ? seg.end : seg.start;

			// 如果回到起点，我们找到了一个闭合的多边形
			if (nextPoint == startPoint)
			{
				currentVertices.push_back(nextPoint); // 添加最后一个点闭合多边形
				foundCycle = true;
				break;
			}

			bool foundNext = false;

			int idx = binary_search(segments, 0, num_segments - 1, nextPoint);
			if (idx != -1)
			{
				PartitionStatus st0 = status[idx], st1 = status[idx + 1];

				if (st0 == 2 || st1 == 2)
				{
					currentSegIdx = (st0 == 2) ? idx : idx + 1;
				}
				else if (st0 == 1 || st1 == 1)
				{
					currentSegIdx = (st0 == 1) ? idx : idx + 1;
				}

				// if (st0 == 1 || st1 == 1)
				// {
				// 	currentSegIdx = (st0 == 1) ? idx : idx + 1;
				// }
				// else if (st0 == 2 || st1 == 2)
				// {
				// 	currentSegIdx = (st0 == 2) ? idx : idx + 1;
				// }
				currentPoint = nextPoint;
				foundNext = true;
			}

			// // 如果回到起点，我们找到了一个闭合的多边形
			// if (!foundNext && nextPoint == startPoint) {
			//     currentVertices.push_back(nextPoint); // 添加最后一个点闭合多边形
			//     foundCycle = true;
			//     break;
			// }

			// 如果没有找到下一个segment，则路径不能闭合
			if (!foundNext)
				break;
		}

		// 如果找到了闭合的多边形并且至少有3个点，添加到结果中
		if (foundCycle && currentVertices.size() >= 3)
		{
			VertexSequence *vs = new VertexSequence(currentVertices.size(), currentVertices.data());
			MyPolygon *currentPolygon = new MyPolygon();
			currentPolygon->set_boundary(vs);
			ctx->intersection_polygons.push_back(currentPolygon);
		}
	}
	return;
}

double Ideal::get_possible_min(Point &p, int center, int step, bool geography)
{
	int core_x_low = get_x(center);
	int core_x_high = get_x(center);
	int core_y_low = get_y(center);
	int core_y_high = get_y(center);

	vector<int> needprocess;

	int ymin = max(0, core_y_low - step);
	int ymax = min(dimy, core_y_high + step);

	double mindist = DBL_MAX;
	// left scan
	if (core_x_low - step >= 0)
	{
		double x = get_pixel_box(core_x_low - step, ymin).high[0];
		double y1 = get_pixel_box(core_x_low - step, ymin).low[1];
		double y2 = get_pixel_box(core_x_low - step, ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// right scan
	if (core_x_high + step <= get_dimx())
	{
		double x = get_pixel_box(core_x_high + step, ymin).low[0];
		double y1 = get_pixel_box(core_x_high + step, ymin).low[1];
		double y2 = get_pixel_box(core_x_high + step, ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0, core_x_low - step + (core_x_low - step >= 0));
	// skip the last if there is right scan
	int xmax = min(dimx, core_x_high + step - (core_x_high + step <= dimx));
	// bottom scan
	if (core_y_low - step >= 0)
	{
		double y = get_pixel_box(xmin, core_y_low - step).high[1];
		double x1 = get_pixel_box(xmin, core_y_low - step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low - step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// top scan
	if (core_y_high + step <= get_dimy())
	{
		double y = get_pixel_box(xmin, core_y_low + step).low[1];
		double x1 = get_pixel_box(xmin, core_y_low + step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low + step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::get_possible_min(box *t_mbr, int core_x_low, int core_y_low, int core_x_high, int core_y_high, int step, bool geography)
{

	vector<int> needprocess;

	int ymin = max(0, core_y_low - step);
	int ymax = min(dimy, core_y_high + step);

	double mindist = DBL_MAX;
	// left scan
	if (core_x_low - step >= 0)
	{
		double x = get_pixel_box(core_x_low - step, ymin).high[0];
		double y1 = get_pixel_box(core_x_low - step, ymin).low[1];
		double y2 = get_pixel_box(core_x_low - step, ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// right scan
	if (core_x_high + step <= get_dimx())
	{
		double x = get_pixel_box(core_x_high + step, ymin).low[0];
		double y1 = get_pixel_box(core_x_high + step, ymin).low[1];
		double y2 = get_pixel_box(core_x_high + step, ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0, core_x_low - step + (core_x_low - step >= 0));
	// skip the last if there is right scan
	int xmax = min(dimx, core_x_high + step - (core_x_high + step <= dimx));
	// bottom scan
	if (core_y_low - step >= 0)
	{
		double y = get_pixel_box(xmin, core_y_low - step).high[1];
		double x1 = get_pixel_box(xmin, core_y_low - step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low - step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// top scan
	if (core_y_high + step <= get_dimy())
	{
		double y = get_pixel_box(xmin, core_y_low + step).low[1];
		double x1 = get_pixel_box(xmin, core_y_low + step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low + step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::distance(Point &p, query_context *ctx, bool profile)
{
	// distance is 0 if contained by the polygon
	double mindist = getMBB()->max_distance(p, ctx->geography);

	bool contained = contain(p, ctx, profile);
	if (contained)
	{
		return 0;
	}

	double mbrdist = mbr->distance(p, ctx->geography);

	// initialize the starting pixel
	int closest = get_closest_pixel(p);

	int step = 0;
	double step_size = get_step(ctx->geography);
	vector<int> needprocess;

	while (true)
	{
		if (step == 0)
		{
			needprocess.push_back(closest);
		}
		else
		{
			needprocess = expand_radius(closest, step);
		}
		// should never happen
		// all the boxes are scanned
		if (needprocess.size() == 0)
		{
			assert(false && "should not evaluated all boxes");
			return boundary->distance(p, ctx->geography);
		}
		for (auto cur : needprocess)
		{
			// printf("checking pixel %d %d %d\n",cur->id[0],cur->id[1],cur->status);
			if (show_status(cur) == BORDER)
			{
				box cur_box = get_pixel_box(get_x(cur), get_y(cur));
				// printf("BOX: lowx=%lf, lowy=%lf, highx=%lf, highy=%lf\n", cur_box.low[0], cur_box.low[1], cur_box.high[0], cur_box.high[1]);
				double mbr_dist = cur_box.distance(p, ctx->geography);
				// skip the pixels that is further than the current minimum
				if (mbr_dist >= mindist)
				{
					continue;
				}

				// the vector model need be checked.

				for (int i = 0; i < get_num_sequences(cur); i++)
				{
					auto rg = get_edge_sequence(get_offset(cur) + i);
					for (int j = 0; j < rg.second; j++)
					{
						auto r = rg.first + j;
						double dist = point_to_segment_distance(p, *get_point(r), *get_point(r + 1), ctx->geography);
						mindist = min(mindist, dist);
						if (ctx->within(mindist))
						{
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();

		// for within query, return if the current minimum is close enough
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = get_possible_min(p, closest, step, ctx->geography);
		// close enough
		if (mindist < minrasterdist)
		{
			break;
		}
	}
	// IDEAL return
	return mindist;
}

// get the distance from pixel pix to polygon target
double Ideal::distance(Ideal *target, int pix, query_context *ctx, bool profile)
{
	assert(show_status(pix) == BORDER);

	auto pix_x = get_x(pix);
	auto pix_y = get_y(pix);
	auto pix_box = get_pixel_box(pix_x, pix_y);
	double mindist = getMBB()->max_distance(pix_box, ctx->geography);
	double mbrdist = getMBB()->distance(pix_box, ctx->geography);
	int step = 0;

	// initialize the seed closest pixels
	vector<int> needprocess = target->get_closest_pixels(pix_box);
	assert(needprocess.size() > 0);
	unsigned short lowx = target->get_x(needprocess[0]);
	unsigned short highx = target->get_x(needprocess[0]);
	unsigned short lowy = target->get_y(needprocess[0]);
	unsigned short highy = target->get_y(needprocess[0]);
	for (auto p : needprocess)
	{
		lowx = min(lowx, (unsigned short)target->get_x(p));
		highx = max(highx, (unsigned short)target->get_x(p));
		lowy = min(lowy, (unsigned short)target->get_y(p));
		highy = max(highy, (unsigned short)target->get_y(p));
	}

	while (true)
	{
		// for later steps, expand the circle to involve more pixels
		if (step > 0)
		{
			needprocess = target->expand_radius(lowx, highx, lowy, highy, step);
		}

		// all the boxes are scanned (should never happen)
		if (needprocess.size() == 0)
		{
			return mindist;
		}

		for (auto cur : needprocess)
		{
			// note that there is no need to check the edges of
			// this pixel if it is too far from the target
			auto cur_x = target->get_x(cur);
			auto cur_y = target->get_y(cur);

			if (target->show_status(cur) == BORDER)
			{
				bool toofar = (target->get_pixel_box(cur_x, cur_y).distance(pix_box, ctx->geography) >= mindist);
				if (toofar)
				{
					continue;
				}
				// the vector model need be checked.
				for (int i = 0; i < get_num_sequences(pix); i++)
				{
					auto pix_er = get_edge_sequence(get_offset(pix) + i);
					for (int j = 0; j < target->get_num_sequences(cur); j++)
					{
						auto cur_er = target->get_edge_sequence(target->get_offset(cur) + j);
						if (cur_er.second < 2 || pix_er.second < 2)
							continue;
						double dist;
						if (ctx->is_within_query())
						{
							dist = segment_to_segment_within_batch(target->boundary->p + cur_er.first,
																   boundary->p + pix_er.first, cur_er.second, pix_er.second,
																   ctx->within_distance, ctx->geography, ctx->edge_checked.counter);
						}
						else
						{
							dist = segment_sequence_distance(target->boundary->p + cur_er.first,
															 boundary->p + pix_er.first, cur_er.second, pix_er.second, ctx->geography);
						}
						mindist = min(dist, mindist);
						if (ctx->within(mindist))
						{
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = target->get_possible_min(&pix_box, lowx, lowy, highx, highy, step, ctx->geography);
		if (mindist < minrasterdist)
			break;
	}

	return mindist;
}

bool Ideal::within(Ideal *target, query_context *ctx)
{
	uint s_level = num_layers;
	uint t_level = target->get_num_layers();
	box source_pixel_box, target_pixel_box;

	queue<pair<int, int>> candidate_pairs;
	for(int i = 0; i < layers[0].get_num_pixels(); i ++){
		for(int j = 0; j < target->layers[0].get_num_pixels(); j ++){
			candidate_pairs.push(make_pair(i, j));
		}
	}

	int i = 0, j = 0;
	while(true){
		vector<int> s_pxs, t_pxs;
		bool s_next_layer = false, t_next_layer = false;
		double s_step = layers[i].get_step_x(), t_step = target->get_layers()[j].get_step_x();

		printf("i = %d j = %d %d %d\n", i, j, s_level, t_level);
		if(i < s_level && (s_step >= t_step || j >= t_level)) {
			i ++;
			s_next_layer = true;
		}
		if(j < t_level && (s_step <= t_step || i >= s_level)) {
			j ++;
			t_next_layer = true;
		}

		int size = candidate_pairs.size();
		if(size == 0) break;
		cout << size << endl;
		for(int k = 0; k < size; k ++){
			auto pair = candidate_pairs.front();
			candidate_pairs.pop();
			int s_pix_id = pair.first, t_pix_id = pair.second;
			
			if(s_next_layer){
				source_pixel_box = layers[i - 1].get_pixel_box(layers[i - 1].get_x(s_pix_id), layers[i - 1].get_y(s_pix_id));

				source_pixel_box.low[0] += 1e-6;
				source_pixel_box.low[1] += 1e-6;
				source_pixel_box.high[0] -= 1e-6;
				source_pixel_box.high[1] -= 1e-6;
				
				auto temp = layers[i].retrieve_pixels(&source_pixel_box);
				printf("temp size %d\n", temp.size());
				for(auto p : temp){
					if(layers[i].show_status(p) == BORDER){
						s_pxs.push_back(p);
					}
				}
			}else{
				if(layers[i].show_status(s_pix_id) == BORDER){
					s_pxs.push_back(s_pix_id);
				}
			}

			if(t_next_layer){
				target_pixel_box = target->get_layers()[j - 1].get_pixel_box(target->get_layers()[j - 1].get_x(t_pix_id), target->get_layers()[j - 1].get_y(t_pix_id));

				target_pixel_box.low[0] += 1e-6;
				target_pixel_box.low[1] += 1e-6;
				target_pixel_box.high[0] -= 1e-6;
				target_pixel_box.high[1] -= 1e-6;

				auto temp = target->get_layers()[j].retrieve_pixels(&target_pixel_box);
				printf("temp size %d\n", temp.size());
				for(auto p : temp){
					if(target->get_layers()[j].show_status(p) == BORDER){
						t_pxs.push_back(p);
					} 
				}
			}else{
				if(target->get_layers()[j].show_status(t_pix_id) == BORDER){
					t_pxs.push_back(t_pix_id);
				}
			}
		}

		cout << candidate_pairs.size() << endl;

		// printf("%d %d\n", s_pxs.size(), t_pxs.size());

		float max_box_dist = 100000.0;
		printf("pxs size %d %d\n", s_pxs.size(), t_pxs.size());
		cout << s_pxs.size() << " !" << t_pxs.size() << endl;
		for(auto id1 : s_pxs){
			auto box1 = layers[i].get_pixel_box(layers[i].get_x(id1), layers[i].get_y(id1));
			for(auto id2 : t_pxs){
				auto box2 = target->get_layers()[j].get_pixel_box(target->get_layers()[j].get_x(id2), target->get_layers()[j].get_y(id2));
				// box1.print();
				// box2.print();
				float max_distacne = box1.max_distance(box2, true);
				if(max_distacne <= ctx->within_distance) return true;
				max_box_dist = min(max_box_dist, max_distacne);
			}
		}

		for(auto id1 : s_pxs){
			auto box1 = layers[i].get_pixel_box(layers[i].get_x(id1), layers[i].get_y(id1));
			for(auto id2 : t_pxs){
				auto box2 = target->get_layers()[j].get_pixel_box(target->get_layers()[j].get_x(id2), target->get_layers()[j].get_y(id2));
				
				float min_distance = box1.distance(box2, true);
				if(min_distance > ctx->within_distance) continue;
				if(min_distance < max_box_dist) {
					candidate_pairs.push(make_pair(id1, id2));
				}
			}
		}
		if(!s_next_layer && !t_next_layer) break;
	}
	printf("result %d\n", candidate_pairs.size());
	ctx->found += candidate_pairs.size();
	return 0;
}