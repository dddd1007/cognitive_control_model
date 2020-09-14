#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *SR_Learning(PyObject *self, PyObject *args)
{
	PyObject *parameters, *stimPosit, *respPosit, *correction;
	char *model_type;
    double alpha=0, beta=0, decay=0, alpha_error=0, ccc=0, alpha_ccc=0, beta_ccc=0;

    if (!PyArg_ParseTuple(args, "sOOOO", &model_type, &parameters, &stimPosit, &respPosit, &correction))
        return NULL;

	Py_ssize_t para_len = PyList_Size(parameters);

    if (!strcmp(model_type, "SR_Q") && para_len == 2) //Q-learning: alpha, beta
    {
        alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
        beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
    }
    else if (!strcmp(model_type, "SR_Q_D") && para_len == 3) //Q-learning with forgetting: alpha, beta, decay
    {
        alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
        beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
        decay = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
    }
    else if (!strcmp(model_type, "SR_Q_D_E") && para_len == 4) //Q-learning with forgetting and error: alpha, beta, decay, alpha_error
    {
        alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
        beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
        decay = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
        alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
    }
    else if (!strcmp(model_type, "SR_Q_D_E_alphaCCC") && para_len == 6) //Q-learning with forgetting, error and cognitive control:alpha, beta, decay, alpha_error, ccc, alpha_ccc
     {  

        alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
        beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
        decay = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
        alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
        ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
		alpha_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 5));
    }
	else if (!strcmp(model_type, "SR_Q_D_E_betaCCC") && para_len == 6) //Q-learning with forgetting, error and cognitive control:alpha, beta, decay, alpha_error, ccc, beta_ccc
	{

		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		decay = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
		ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
		beta_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 5));
	}
	else if (!strcmp(model_type, "SR_Q_D_E_alphaCCC_betaCCC") && para_len == 7) //Q-learning with forgetting, error and cognitive control:alpha, beta, decay, alpha_error, ccc, alpha_ccc, beta_ccc
	{
		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		decay = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
		ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
		alpha_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 5));
		beta_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 6));
	}
     else
     {   //return error information
		//PyErr_SetString(PyExc_RuntimeError, "wrong number of model parameters");
		return NULL;
     }
	
    Py_ssize_t trial_num = PyList_Size(stimPosit);
	
	double P=0, pe[2][2] = { 0,0,0,0 }, Q[2][2] = { 0.5,0.5,0.5,0.5 };
	double alpha_temp = 0, beta_temp = beta, conflict = 0;
    long i_stimPosit, i_respPosit, i_correction, i_oppAction, i_oppstimPosit;
	long L_one = 1;
    int i;
	
    PyObject *P_current, *Q_current, *pe_current;
	PyObject *P_list = PyList_New(trial_num);
	PyObject *Q_list = PyList_New(trial_num);
	PyObject *pe_list = PyList_New(trial_num);
		
    for (i = 0; i < trial_num; i++)
    {
        i_stimPosit = PyLong_AsLong(PyList_GetItem(stimPosit, i));
        i_respPosit = PyLong_AsLong(PyList_GetItem(respPosit, i));
        i_correction = PyLong_AsLong(PyList_GetItem(correction, i));
		i_oppAction = L_one - i_respPosit;

        P = exp(beta_temp * Q[i_stimPosit][i_respPosit])/(exp(beta_temp * Q[i_stimPosit][i_respPosit]) + exp(beta_temp * Q[i_stimPosit][i_oppAction]));

        //store P, Q
		P_current = PyFloat_FromDouble(P);		
        Q_current = Py_BuildValue("[[d,d],[d,d]]", Q[0][0],Q[0][1],Q[1][0],Q[1][1]);
        
		PyList_SetItem(P_list, i, P_current);
        PyList_SetItem(Q_list, i, Q_current);

        //update Q
        if (para_len < 4) //
        {
            alpha_temp = alpha;
        }
        else if (para_len == 4)
        {
            if (i_correction == 1)
                alpha_temp = alpha;
            else
                alpha_temp = alpha_error;
        }
		else if (!strcmp(model_type, "SR_Q_D_E_alphaCCC") && para_len == 6)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{			
				if (conflict < ccc)
					alpha_temp = alpha_ccc;
				else
					alpha_temp = alpha;
			}
			else
			{
				if (-conflict < ccc)
					alpha_temp = alpha_ccc;
				else
					alpha_temp = alpha_error;
			}
		}
		else if (!strcmp(model_type, "SR_Q_D_E_betaCCC") && para_len == 6)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{
				alpha_temp = alpha;

				if (conflict < ccc)
					beta_temp = beta_ccc;
				else
					beta_temp = beta;
				
			}
			else
			{
				alpha_temp = alpha_error;

				if (-conflict < ccc)
					beta_temp = beta_ccc;
				else
					beta_temp = beta;
			}
			

		}
		else if (para_len == 7)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{
				if (conflict < ccc)
				{
					alpha_temp = alpha_ccc;
					beta_temp = beta_ccc;
				}	
				else
				{
					alpha_temp = alpha;
					beta_temp = beta;
				}
					
			}
			else
			{
				if (-conflict < ccc)
				{
					alpha_temp = alpha_ccc;
					beta_temp = beta_ccc;
				}
				else
				{
					alpha_temp = alpha_error;
					beta_temp = beta;
				}
			}
		}

        //update Q
        memset(pe, 0, sizeof(pe));
        pe[i_stimPosit][i_respPosit] = i_correction - Q[i_stimPosit][i_respPosit];
        Q[i_stimPosit][i_respPosit] = Q[i_stimPosit][i_respPosit] + alpha_temp * pe[i_stimPosit][i_respPosit];

        pe[i_stimPosit][i_oppAction] = (L_one - i_correction) - Q[i_stimPosit][i_oppAction];
        Q[i_stimPosit][i_oppAction] = Q[i_stimPosit][i_oppAction] + alpha_temp * pe[i_stimPosit][i_oppAction];

		//store pe
		pe_current = Py_BuildValue("[[d,d],[d,d]]", pe[0][0], pe[0][1], pe[1][0], pe[1][1]);
		PyList_SetItem(pe_list, i, pe_current);

        //decay
		if (para_len > 2)
        {
            i_oppstimPosit = L_one - i_stimPosit;
            Q[i_oppstimPosit][0] = Q[i_oppstimPosit][0] + decay * (0.5 - Q[i_oppstimPosit][0]);
            Q[i_oppstimPosit][1] = Q[i_oppstimPosit][1] + decay * (0.5 - Q[i_oppstimPosit][1]);
        }
		
    }
	
	PyObject *result = Py_BuildValue("OOO", P_list, Q_list, pe_list);

	//如果下面不释放内存的话会造成内存泄漏；内存会占用的越来越大
	Py_DECREF(P_list); 
	Py_DECREF(Q_list);
	Py_DECREF(pe_list);

	return result;
}

// Abstract incompatible learning
static PyObject* AB_Learning(PyObject* self, PyObject* args)
{
	PyObject *parameters, *congruency, *correction;
	char *model_type;
	double alpha = 0, beta = 0, alpha_error = 0, ccc = 0, alpha_ccc = 0, beta_ccc = 0;

	if (!PyArg_ParseTuple(args, "sOOO", &model_type, &parameters, &congruency, &correction))
		return NULL;

	Py_ssize_t para_len = PyList_Size(parameters);

	if (!strcmp(model_type, "AB_Q") && para_len == 2) //Q-learning: alpha, beta
	{
		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
	}
	else if (!strcmp(model_type, "AB_Q_E") && para_len == 3) //Q-learning with error: alpha, beta, alpha_error
	{
		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
	}
	else if (!strcmp(model_type, "AB_Q_E_alphaCCC") && para_len == 5) //Q-learning with error and cognitive control:alpha, beta, alpha_error, ccc, alpha_ccc
	{

		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
		ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
		alpha_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
	}
	else if (!strcmp(model_type, "AB_Q_E_betaCCC") && para_len == 5) //Q-learning with error and cognitive control:alpha, beta, alpha_error, ccc, beta_ccc
	{

		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
		ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
		beta_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
	}
	else if (!strcmp(model_type, "AB_Q_E_alphaCCC_betaCCC") && para_len == 6) //Q-learning with error and cognitive control:alpha, beta, alpha_error, ccc, alpha_ccc, beta_ccc
	{
		alpha = PyFloat_AsDouble(PyList_GetItem(parameters, 0));
		beta = PyFloat_AsDouble(PyList_GetItem(parameters, 1));
		alpha_error = PyFloat_AsDouble(PyList_GetItem(parameters, 2));
		ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 3));
		alpha_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 4));
		beta_ccc = PyFloat_AsDouble(PyList_GetItem(parameters, 5));
	}
	else
	{   //return error information
	   //PyErr_SetString(PyExc_RuntimeError, "wrong number of model parameters");
		return NULL;
	}

	Py_ssize_t trial_num = PyList_Size(congruency);

	double P=0, pe=0, Q = 0.5;  //Q is for the incongruent condition.
	double alpha_temp = 0, beta_temp = beta, conflict = 0;
	long i_congruency, i_correction;
	long L_one = 1;
	int i;

	PyObject* P_current, * Q_current, * pe_current;
	PyObject* P_list = PyList_New(trial_num);
	PyObject* Q_list = PyList_New(trial_num);
	PyObject* pe_list = PyList_New(trial_num);

	for (i = 0; i < trial_num; i++)
	{
		i_congruency = PyLong_AsLong(PyList_GetItem(congruency, i)); //0, congruent; 1, incongruent
		i_correction = PyLong_AsLong(PyList_GetItem(correction, i)); //0, error; 1, correct

		if (i_congruency == 1)
			P = exp(beta_temp * Q) / (exp(beta_temp * Q) + exp(beta_temp * (1 - Q)));
		else
			P = exp(beta_temp * (1 - Q)) / (exp(beta_temp * Q) + exp(beta_temp * (1 - Q)));

		//store P, Q
		P_current = PyFloat_FromDouble(P);
		Q_current = PyFloat_FromDouble(Q);

		PyList_SetItem(P_list, i, P_current);
		PyList_SetItem(Q_list, i, Q_current);

		//update Q
		if (para_len < 3) //
		{
			alpha_temp = alpha;
		}
		else if (para_len == 3)
		{
			if (i_correction == 1)
				alpha_temp = alpha;
			else
				alpha_temp = alpha_error;
		}
		else if (!strcmp(model_type, "AB_Q_E_alphaCCC") && para_len == 5)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{
				if (conflict < ccc)
					alpha_temp = alpha_ccc;
				else
					alpha_temp = alpha;
			}
			else
			{
				if (-conflict < ccc)
					alpha_temp = alpha_ccc;
				else
					alpha_temp = alpha_error;
			}
		}
		else if (!strcmp(model_type, "AB_Q_E_betaCCC") && para_len == 5)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{
				alpha_temp = alpha;

				if (conflict < ccc)
					beta_temp = beta_ccc;
				else
					beta_temp = beta;
			}
			else
			{
				alpha_temp = alpha_error;

				if (-conflict < ccc)
					beta_temp = beta_ccc;
				else
					beta_temp = beta;
			}

			

		}
		else if (para_len == 6)
		{
			conflict = 2 * P - 1;

			if (i_correction == 1)
			{
				if (conflict < ccc)
				{
					alpha_temp = alpha_ccc;
					beta_temp = beta_ccc;
				}
				else
				{
					alpha_temp = alpha;
					beta_temp = beta;
				}

			}
			else
			{
				if (-conflict < ccc)
				{
					alpha_temp = alpha_ccc;
					beta_temp = beta_ccc;
				}
				else
				{
					alpha_temp = alpha_error;
					beta_temp = beta;
				}
			}
		}

		//update Q
		if (i_congruency == 1)
		{
			pe = i_correction - Q;
			Q = Q + alpha_temp * pe;
		}
		else if (i_congruency == 0)
		{
			pe = (1 - i_correction) - Q;
			Q = Q + alpha_temp * pe;
		}
		else
		{
			return NULL;
		}
		

		//store pe
		pe_current = PyFloat_FromDouble(pe);
		PyList_SetItem(pe_list, i, pe_current);

	}

	PyObject* result = Py_BuildValue("OOO", P_list, Q_list, pe_list);

	//如果下面不释放内存的话会造成内存泄漏；内存会占用的越来越大
	Py_DECREF(P_list);
	Py_DECREF(Q_list);
	Py_DECREF(pe_list);

	return result;
}



static char sr_docs[] = "SR_Learning(model_type, parameters, stimPosit, respPosit, correction): five input arguments\n";
static char ab_docs[] = "AB_Learning(model_type, parameters, congruency, correction): four input arguments\n";

static PyMethodDef rlcc_funcs[] = {
    {"SR_Learning", (PyCFunction)SR_Learning, METH_VARARGS, sr_docs},
	{"AB_Learning", (PyCFunction)AB_Learning, METH_VARARGS, ab_docs},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef RLCC = {
    PyModuleDef_HEAD_INIT,
    "RLCC",
    "",
    -1,
    rlcc_funcs};

PyMODINIT_FUNC PyInit_RLCC(void)
{
    return PyModule_Create(&RLCC);
}