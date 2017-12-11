#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define count 11 //giju skaicius
#define masyvuCount 5	//masyvu skaicius

struct Item
{
	char name[20 * count];
	int amount;
	int eilNr; // eiles numeris masyve
	double price;
};

struct Supplier {
	struct Item items[count];
	int itemCount;
	char supplierName[20];
};

//tiekeju masyvas
struct Supplier suppliers[masyvuCount];

//bendras masyvas
struct Item bendras[count];

void readFile();
void printFile(struct Item supplier[], int  itemCount);
void printSuppliers(struct Supplier suppliers[]);
void printItemsArray(struct Item supplier[], int itemCount);
void printItemsResults(struct Item supplier[], int  itemCount);

__device__ void addElements(struct Item bendras[], struct Supplier duom[], int id);
__device__ void addElement(struct Item bendras[], struct Item element, int index);
__device__ char * cuda_strcpy(char *dest, const char *src);
__device__ char * cuda_strcat(char *dest, const char *src);

/** Pagrindine lygiagreti funkcija
*	@param f - bendras tiekeju masyvas
*	@param bendras - bendras masyvas
*/
__global__ void addKernel(struct Supplier f[], struct Item bendras[])
{
	int tid = threadIdx.x; 			//gijos numeris
	addElements(bendras, f, tid);	//elemento sudejimas i bendra masyva
}

/**Elemento sudejimo funkcija i bendra masyva
*	@param bendras - bendras masyvas
*	@param duom - tiekeju masyvas
*	@param id - gijos numeris, pagal kuria imsim duomenis
*/
__device__ void addElements(struct Item bendras[], struct Supplier duom[], int id) {
	for (int j = 0; j < duom[0].itemCount; j++) {
		addElement(bendras, duom[j].items[id], id);
	}
}

/**Vieno elemento pridejimas
*	@param bendras	- bendras masyvas
*	@param element - preke
*	@param index - bendro masyvo indeksas
*/
__device__ void addElement(struct Item bendras[], struct Item element, int index) {
	bendras[index].eilNr = index;
	bendras[index].price = bendras[index].price + element.price;
	bendras[index].amount = bendras[index].amount + element.amount;
	cuda_strcat(bendras[index].name, element.name);
}

/** Stringo kopijavimas
*	@param dest - kur talpinsime
*	@param src - ka talpinsime
*/
__device__ char * cuda_strcpy(char *dest, const char *src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
}

/*	Stringo kopijavimas
*	@param dest - kur talpinsime
*	@param src - ka talpinsime
*/
__device__ char * cuda_strcat(char *dest, const char *src) {
	int i = 0;
	while (dest[i] != 0) i++;
	cuda_strcpy(dest + i, src);
	return dest;
}

int main(int argc, char *argv[])
{
	readFile();
	// pradiniu duomenu spausdinimas
	printSuppliers(suppliers);
	// tiekejai
	struct Supplier *s;
	// bendras
	struct Item *bendrasCuda;

	//atminties isskyrimas GPU
	cudaMalloc((void**)&s, sizeof(Supplier)*masyvuCount);
	// kopijavimas i GPU
	cudaMemcpy(s, suppliers, sizeof(Supplier)*masyvuCount, cudaMemcpyHostToDevice);
	//atminities isskyrimas i GPU
	cudaMalloc((void**)&bendrasCuda, sizeof(Item)*count);
	//kopijavimas i GPU
	cudaMemcpy(bendrasCuda, bendras, sizeof(Item)*count, cudaMemcpyHostToDevice);

	// giju skaiciaus parinkimas ir lygiagretaus kodo startavimas
	addKernel << < 1, count >> >(s, bendrasCuda);
	
	// kopijavimas i CPU
	cudaMemcpy(bendras, bendrasCuda, sizeof(Item)*count, cudaMemcpyDeviceToHost);
	//atlaisvinimas atminties is GPU	
	cudaFree((void**)&s);
	//atlaisvinimas atminties is GPU
	cudaFree((void**)&bendrasCuda);

	printItemsResults(bendras, count);
	printFile(bendras, count);
	printf("Done \n");
	return 0;
}


/**
*	@param suppliers - tiekeju masyvas
*/
void printSuppliers(struct Supplier suppliers[]) {
	for (int i = 0; i < masyvuCount; i++)
	{
		printf("*** %s *** \n", suppliers[i].supplierName);
		printf("  %-10s %-10s %-4s \n", "Name", "Amount", "Price");
		printItemsArray(suppliers[i].items, suppliers[i].itemCount);
	}
}

void printItemsArray(struct Item supplier[], int  itemCount) {
	for (int i = 0; i < itemCount; i++)
	{
		if (supplier[i].name) {
			printf("%d %-10s %-10d %-4.2f \n", supplier[i].eilNr, 
				supplier[i].name, supplier[i].amount, supplier[i].price);
		}
	}
}


/**
* Rezultatu spausdinimas
* @param supplier - tiekejo masyvas
*/
void printItemsResults(struct Item supplier[], int  itemCount) {
	printf("\nREZULTATAI \n\n");
	for (int i = 0; i < itemCount; i++)
	{
		if (supplier[i].name) {
			printf("%d %-50s %-10d %-4.2f \n", supplier[i].eilNr, supplier[i].name, supplier[i].amount, supplier[i].price);
		}
	}
}

/**
*	Skaitymas is failo
*/
void readFile() {
	errno_t err;
	FILE *stream;
	char file_name[21] = "SankauskasS_L4.txt";  //failo vardas
	err = fopen_s(&stream, file_name, "r");
	
	char name[20];
	int n; //prekiu skaicius
	int supplierCount = 0; //tiekejo iteracijos kintamasis
	
	while (true) {
		int readItems = fscanf(stream, "%s %d", name, &n);
		if (readItems == 2) {
			strcpy(suppliers[supplierCount].supplierName, name);
			suppliers[supplierCount].itemCount = n;
			for (int i = 0; i < n; i++) {
				struct Item item = suppliers[supplierCount].items[i];
				char item_name[20];
				int amount;
				double price;
				fscanf(stream, "%s %d %lf", item_name, &amount, &price);
				suppliers[supplierCount].items[i].amount = amount;
				suppliers[supplierCount].items[i].eilNr = i;
				suppliers[supplierCount].items[i].price = price;
				strcpy(suppliers[supplierCount].items[i].name, item_name);
			}
			supplierCount++;
		}
		else if (readItems == EOF) {
			break;
		}
	}
	if (stream)
		err = fclose(stream);
}

/**
 * Rasymas i faila
 */
void printFile(struct Item supplier[], int  itemCount)
{
	FILE *f = fopen("SankauskasS_L4a_rez.txt", "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f,"\nREZULTATAI \n\n");
	fprintf(f, "%-3s %-50s %-10s %-7s \n", "Nr", "Name", "Amount", "Price");
	for (int i = 0; i < itemCount; i++)
	{
		if (supplier[i].name) {
			fprintf(f, "%-3d %-50s %-10d %-7.2f \n", supplier[i].eilNr, 
				supplier[i].name, supplier[i].amount, supplier[i].price);
		}
	}

	fclose(f);
}
