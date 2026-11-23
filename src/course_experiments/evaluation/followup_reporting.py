from __future__ import annotations
import random
import math


def paired(left,right,seed=42,replicates=2000):
    if set(left)!=set(right):raise ValueError('Paired comparison requires same fixed questions')
    values=[left[k]-right[k] for k in sorted(left)];n=len(values)
    if not n:return dict(paired_questions=0,mean_difference=None,percentile_interval_95=None)
    rng=random.Random(seed);samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(replicates))
    return dict(paired_questions=n,mean_difference=sum(values)/n,percentile_interval_95=[samples[int(.025*replicates)],samples[min(replicates-1,int(.975*replicates))]],
        bootstrap_unit='question within cohort; not model-seed variability',replicates=replicates,seed=seed)


def paired_course_macro(left,right,book_courses,seed=42,replicates=2000):
    if set(left)!=set(right) or set(left)!=set(book_courses) or not left:
        raise ValueError('Both methods must cover the exact declared book collection')
    if type(replicates) is not int or replicates<100:raise ValueError('At least100 bootstrap replicates required')
    seen=set();differences={};counts={};empty=[];courses={}
    for book in sorted(book_courses):
        if set(left[book])!=set(right[book]):raise ValueError('Paired methods require the same fixed question IDs per book')
        ids=sorted(left[book]);counts[book]=len(ids)
        if seen.intersection(ids):raise ValueError('Repeated questions across books cannot be treated as new samples')
        seen.update(ids);courses.setdefault(book_courses[book],[]).append(book)
        for values in [left[book],right[book]]:
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values.values()):
                raise ValueError('Finite normalized scores required; real failed questions retain explicit zero')
        if ids:differences[book]=sum(left[book][q]-right[book][q] for q in ids)/len(ids)
        else:empty.append(book)
    result=dict(complete_book_coverage=not empty,empty_books=empty,paired_questions=len(seen),book_question_counts=counts,
        book_count=len(book_courses),course_count=len(courses),book_difference=differences,
        weighting='equal_books_within_each_course_then_equal_courses',
        bootstrap_unit='paired_whole_book_within_each_fixed_course',replicates=replicates,seed=seed,
        limitation='conditional_on_these_fixed_courses_and_graphs;few_books;not_model_seed_variability_or_generalization_to_new_courses',
        mean_difference=None,course_difference=None,percentile_interval_95=None)
    if empty:return result
    course_difference={course:sum(differences[b] for b in books)/len(books) for course,books in sorted(courses.items())}
    rng=random.Random(seed);samples=[]
    for _ in range(replicates):
        # 两方法共用整书抽样；不把四书课程、重复cohort或书内题量当独立优势。
        values=[sum(differences[books[rng.randrange(len(books))]] for _ in books)/len(books) for _,books in sorted(courses.items())]
        samples.append(sum(values)/len(values))
    samples.sort();result.update(mean_difference=sum(course_difference.values())/len(course_difference),course_difference=course_difference,
        percentile_interval_95=[samples[int(.025*replicates)],samples[min(replicates-1,int(.975*replicates))]])
    return result


def holm_adjust(pvalues):
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in pvalues.values()):
        raise ValueError('Finite probability values required')
    ordered=sorted(pvalues,key=lambda key:(pvalues[key],key));adjusted={};previous=0.0
    for index,key in enumerate(ordered):
        previous=max(previous,min(1.0,pvalues[key]*(len(ordered)-index)));adjusted[key]=previous
    return {key:adjusted[key] for key in pvalues}
